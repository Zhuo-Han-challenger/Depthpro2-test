"""Estimate multi-view camera poses and write a manifest for export-colmap.py."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap

from rgb_images import read_rgb, supported_image_suffixes


def discover_inputs(image_dir, depth_dir):
    if not image_dir.is_dir():
        raise NotADirectoryError(f"RGB directory does not exist: {image_dir}")
    if not depth_dir.is_dir():
        raise NotADirectoryError(f"Depth Pro 2 output directory does not exist: {depth_dir}")
    files = sorted(
        path for path in image_dir.rglob("*") if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(image_dir).parts)
    )
    suffixes = supported_image_suffixes()
    sources = [path for path in files if path.suffix.lower() in suffixes]
    if len(sources) < 2:
        example_files = ", ".join(str(path.relative_to(image_dir)) for path in files[:8])
        raise ValueError(
            f"found {len(sources)} supported RGB images under {image_dir} (including subfolders). "
            f"At least two are required. Files seen: {example_files or '(none)'}."
        )
    pairs = [(source, depth_dir / f"{source.stem}-depth.npz") for source in sources]
    if len({source.stem.casefold() for source in sources}) != len(sources):
        raise ValueError("RGB basenames must be unique across subfolders to match Depth Pro 2 NPZs")
    missing = [f"{source.relative_to(image_dir)} -> {depth.name}" for source, depth in pairs
               if not depth.is_file()]
    if missing:
        raise ValueError(
            f"found {len(sources)} RGB images, but {len(missing)} have no matching NPZ in {depth_dir}: "
            + ", ".join(missing[:8])
        )
    return pairs


def prepare_images(image_dir, pairs, prepared_dir, database, threads):
    extraction = pycolmap.FeatureExtractionOptions(num_threads=threads)
    frames = {}
    for index, (source, depth_path) in enumerate(pairs):
        with read_rgb(source) as rgb, np.load(depth_path, allow_pickle=False) as prediction:
            width, height = rgb.size
            focal = float(prediction["focallength_px"].item())
            if prediction["depth"].shape != (height, width):
                raise ValueError(f"{source.name}: depth shape must match the oriented RGB image")
            if not np.isfinite(focal) or focal <= 0:
                raise ValueError(f"{depth_path.name}: expected a positive focal length")

            # COLMAP and the exporter see the same pixel grid even when EXIF rotates the source.
            name = f"{index:06d}.bmp"
            rgb.save(prepared_dir / name)
        intrinsics = [focal, focal, width / 2, height / 2]
        pycolmap.extract_features(
            database,
            prepared_dir,
            image_names=[name],
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            reader_options=pycolmap.ImageReaderOptions(
                camera_model="PINHOLE", camera_params=",".join(map(str, intrinsics))
            ),
            extraction_options=extraction,
            device=pycolmap.Device.cpu,
        )
        frames[name] = {
            "image": str(source), "depth": str(depth_path),
            "name": source.relative_to(image_dir).as_posix(),
        }
    return frames


def recover_reconstruction(database, prepared_dir, model_dir, frames, threads):
    pycolmap.match_exhaustive(
        database,
        matching_options=pycolmap.FeatureMatchingOptions(num_threads=threads),
        device=pycolmap.Device.cpu,
    )
    options = pycolmap.IncrementalPipelineOptions(
        num_threads=threads, min_model_size=2, ba_refine_focal_length=False
    )
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.init_min_tri_angle = 4.0
    options.triangulation.ignore_two_view_tracks = False
    def map_images(output):
        return pycolmap.incremental_mapping(database, prepared_dir, output, options=options)

    best_model = None
    best_registered = {}
    def keep_model(model):
        nonlocal best_model, best_registered
        registered = {model.images[index].name: model.images[index] for index in model.reg_image_ids()}
        if len(registered) > len(best_registered):
            best_model = model
            best_registered = registered

    for model in map_images(model_dir).values():
        keep_model(model)
    if len(best_registered) == len(frames):
        return best_model, best_registered

    with pycolmap.Database.open(database) as db:
        names = {image.image_id: image.name for image in db.read_all_images()}
        pair_ids, counts = db.read_two_view_geometry_num_inliers()
    ranked_pairs = sorted(zip(counts, pair_ids), reverse=True)
    tried = 0
    for count, pair_id in ranked_pairs:
        first, second = pycolmap.pair_id_to_image_pair(pair_id)
        if count < options.min_num_matches or {names[first], names[second]} <= best_registered.keys():
            continue
        print(f"Trying a different COLMAP seed ({count} verified matches): "
              f"{frames[names[first]]['name']} + {frames[names[second]]['name']}", flush=True)
        options.init_image_id1, options.init_image_id2 = first, second
        attempt = model_dir / f"seed-{first}-{second}"
        attempt.mkdir()
        for model in map_images(attempt).values():
            keep_model(model)
        tried += 1
        if len(best_registered) == len(frames) or tried == 3:
            break

    if len(best_registered) < 2:
        raise ValueError("COLMAP could not establish a shared coordinate frame from any image pair")
    return best_model, best_registered


def depth_at(path, pixels):
    x, y = np.floor(pixels).astype(np.int64).T
    with np.load(path, allow_pickle=False) as prediction:
        depth = prediction["depth"]
        valid = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        samples = np.full(len(pixels), np.nan)
        samples[valid] = depth[y[valid], x[valid]]
    return samples


def estimate_metric_scale(model, registered, frames):
    log_ratios = []
    observations = 0
    for name, image in registered.items():
        linked = [point for point in image.points2D if point.has_point3D()]
        if not linked:
            continue
        world_points = np.array([model.points3D[point.point3D_id].xyz for point in linked])
        z = (image.cam_from_world() * world_points)[:, 2]
        depth_samples = depth_at(frames[name]["depth"], np.array([point.xy for point in linked]))
        valid = np.isfinite(z) & (z > 0) & np.isfinite(depth_samples) & (depth_samples > 0)
        if valid.any():
            log_ratios.append(np.log(depth_samples[valid].astype(np.float64)) - np.log(z[valid]))
            observations += int(valid.sum())

    if observations < 10:
        raise ValueError(f"only {observations} valid depth correspondences for estimating metric scale")
    samples = np.concatenate(log_ratios)
    median = np.median(samples)
    scale = float(np.exp(median))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("could not compute a finite positive metric scale")
    median_relative_error = float(np.median(np.abs(np.expm1(median - samples))))
    return scale, observations, median_relative_error


def localize_missing_views(database, model, registered, frames, scale):
    """Localize missing views against metric points anchored by already registered cameras."""
    poses, cameras, diagnostics = {}, {}, {}
    for name, image in registered.items():
        pose = image.cam_from_world()
        poses[name] = pycolmap.Rigid3d(pose.rotation, pose.translation * scale)
        cameras[name] = model.cameras[image.camera_id]
    if len(poses) == len(frames):
        return poses, cameras, diagnostics

    with pycolmap.Database.open(database) as db:
        db_images = {image.name: image for image in db.read_all_images()}
        for name, frame in frames.items():
            if name in registered:
                continue
            target = db_images[name]
            camera = db.read_camera(target.camera_id)
            target_xy = db.read_keypoints(target.image_id)[:, :2].astype(np.float64)
            groups = []
            for ref_name, ref in registered.items():
                if not db.exists_two_view_geometry(ref.image_id, target.image_id):
                    continue
                matches = db.read_two_view_geometry(ref.image_id, target.image_id).inlier_matches
                if not len(matches):
                    continue
                ref_ids, target_ids = matches.T
                ref_xy = db.read_keypoints(ref.image_id)[ref_ids, :2].astype(np.float64)
                z = depth_at(frames[ref_name]["depth"], ref_xy)
                valid_depth = np.isfinite(z) & (z > 0)
                fx, fy, cx, cy = cameras[ref_name].params
                rays = (ref_xy[valid_depth] - [cx, cy]) / [fx, fy]
                points = np.full((len(matches), 3), np.nan)
                points[valid_depth] = poses[ref_name].inverse() * (
                    np.column_stack((rays, np.ones(len(rays)))) * z[valid_depth, None]
                )
                # Prefer triangulated geometry where the reference keypoint has a sparse track.
                for row, ref_id in enumerate(ref_ids):
                    point = ref.points2D[int(ref_id)]
                    if point.has_point3D():
                        points[row] = model.points3D[point.point3D_id].xyz * scale
                valid = np.isfinite(points).all(axis=1)
                if valid.any():
                    groups.append((target_ids[valid], points[valid]))

            estimate = choose_absolute_pose(camera, target_xy, groups)
            if estimate is None:
                count = len(np.unique(np.concatenate([ids for ids, _ in groups]))) if groups else 0
                raise ValueError(
                    f"COLMAP registered {len(registered)}/{len(frames)} views. "
                    f"Cannot reliably localize {frame['name']} from {count} matched keypoints "
                    "with usable 3D geometry. Check overlap on stationary, textured surfaces."
                )
            poses[name], diagnostics[name] = estimate
            cameras[name] = camera
            print(f"Localized {frame['name']} with depth-assisted PnP: "
                  f"{diagnostics[name]['inliers']} inliers", flush=True)
    return poses, cameras, diagnostics


def choose_absolute_pose(camera, target_xy, groups):
    if not groups:
        return None
    ids = np.concatenate([group[0] for group in groups])
    xyz = np.concatenate([group[1] for group in groups])
    _, unique = np.unique(ids, return_index=True)
    if len(unique) < 30:
        return None
    pixels = target_xy[ids]
    max_error = float(np.clip(max(camera.width, camera.height) * 0.003, 3, 12))
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = max_error
    best = None
    for subset_ids, subset_xyz in [*groups, (ids[unique], xyz[unique])]:
        if len(subset_ids) < 30:
            continue
        result = pycolmap.estimate_and_refine_absolute_pose(
            target_xy[subset_ids], subset_xyz, camera, estimation_options=options
        )
        if result is None:
            continue
        pose = result["cam_from_world"]
        camera_xyz = pose * xyz
        reprojection = camera.img_from_cam(camera_xyz, check_cheirality=False)
        errors = np.linalg.norm(reprojection - pixels, axis=1)
        inliers = (camera_xyz[:, 2] > 0) & (errors <= max_error)
        count = len(np.unique(ids[inliers]))
        if count < max(30, 0.25 * len(unique)):
            continue
        median_error = float(np.median(errors[inliers]))
        score = (count, -median_error)
        if best is None or score > best[0]:
            best = (score, pose, {"inliers": count, "median_reprojection_px": median_error})
    return None if best is None else best[1:]


def estimate_poses(image_dir, depth_dir, manifest_path, threads):
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    pairs = discover_inputs(image_dir, depth_dir)
    print(f"Found {len(pairs)} RGB images with matching depth NPZs:", flush=True)
    for source, _ in pairs:
        print(f"  {source.relative_to(image_dir)}", flush=True)

    with TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        database = scratch / "features.db"
        prepared_dir, model_dir = scratch / "images", scratch / "models"
        prepared_dir.mkdir()
        model_dir.mkdir()
        frames = prepare_images(image_dir, pairs, prepared_dir, database, threads)
        model, registered = recover_reconstruction(database, prepared_dir, model_dir, frames, threads)
        scale, observations, depth_error = estimate_metric_scale(model, registered, frames)
        poses, cameras, pnp = localize_missing_views(database, model, registered, frames, scale)
        origin = poses[next(iter(frames))].inverse()
        for name, frame in frames.items():
            frame["intrinsics"] = cameras[name].params.tolist()
            frame["world_to_camera"] = (poses[name] * origin).matrix().tolist()
            frame["pose_source"] = "depth_pnp" if name in pnp else "colmap"
            if name in pnp:
                frame["pnp"] = pnp[name]

    manifest = {
        "frames": list(frames.values()),
        "pose_estimation": {
            "scale_m_per_colmap_unit": scale,
            "depth_observations": observations,
            "median_relative_depth_disagreement": depth_error,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(
        f"Wrote {len(frames)} camera poses to {manifest_path}. "
        f"Depth scale used {observations} observations "
        f"(median depth disagreement {depth_error:.1%})."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="directory of overlapping RGB views")
    parser.add_argument("depth_dir", type=Path, help="directory with matching Depth Pro 2 NPZs")
    parser.add_argument("manifest", type=Path, help="output JSON manifest for export-colmap.py")
    parser.add_argument("--threads", type=int, default=4, help="COLMAP CPU threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        estimate_poses(
            args.image_dir.resolve(), args.depth_dir.resolve(), args.manifest, args.threads
        )
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Pose estimation failed: {exc}\n")


if __name__ == "__main__":
    main()
