"""Estimate multi-view camera poses and write a manifest for export-colmap.py."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_images(image_dir, depth_dir, prepared_dir, database, threads):
    sources = sorted(
        image for image in image_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(sources) < 2:
        raise ValueError("pose estimation requires at least two overlapping RGB images")
    if len({source.stem for source in sources}) != len(sources):
        raise ValueError("RGB basenames must be unique without their extensions")

    extraction = pycolmap.FeatureExtractionOptions(num_threads=threads)
    frames = {}
    for index, source in enumerate(sources):
        depth_path = depth_dir / f"{source.stem}-depth.npz"
        with Image.open(source) as original, np.load(depth_path, allow_pickle=False) as prediction:
            rgb = ImageOps.exif_transpose(original).convert("RGB")
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
        frames[name] = {"image": str(source), "depth": str(depth_path)}
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
    models = pycolmap.incremental_mapping(database, prepared_dir, model_dir, options=options)

    best_registered = {}
    for model in sorted(models.values(), key=lambda candidate: -candidate.num_reg_images()):
        registered = {model.images[index].name: model.images[index] for index in model.reg_image_ids()}
        if set(registered) == set(frames):
            return model, registered
        if len(registered) > len(best_registered):
            best_registered = registered
    missing = ", ".join(Path(frames[name]["image"]).name for name in frames if name not in best_registered)
    raise ValueError(
        f"COLMAP registered {len(best_registered)}/{len(frames)} images in its best model. "
        f"Unregistered: {missing}. Check for shared textured features and camera baseline."
    )


def estimate_metric_scale(model, registered, frames):
    log_ratios = []
    observations = 0
    for name, frame in frames.items():
        image = registered[name]
        linked = [point for point in image.points2D if point.has_point3D()]
        if not linked:
            continue
        pixels = np.floor([point.xy for point in linked]).astype(np.int64)
        world_points = np.array([model.points3D[point.point3D_id].xyz for point in linked])
        z = (image.cam_from_world() * world_points)[:, 2]
        with np.load(frame["depth"], allow_pickle=False) as prediction:
            depth = prediction["depth"]
            x, y = pixels.T
            in_image = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
            depth_samples = depth[y[in_image], x[in_image]]
        z = z[in_image]
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


def estimate_poses(image_dir, depth_dir, manifest_path, threads):
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    with TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        database = scratch / "features.db"
        prepared_dir, model_dir = scratch / "images", scratch / "models"
        prepared_dir.mkdir()
        model_dir.mkdir()
        frames = prepare_images(image_dir, depth_dir, prepared_dir, database, threads)
        model, registered = recover_reconstruction(database, prepared_dir, model_dir, frames, threads)
        scale, observations, depth_error = estimate_metric_scale(model, registered, frames)

        # Express the reconstruction in the first camera's axes and give translations metric scale.
        origin = registered[next(iter(frames))].cam_from_world().inverse()
        for name, frame in frames.items():
            image = registered[name]
            pose = (image.cam_from_world() * origin).matrix()
            pose[:, 3] *= scale
            frame["intrinsics"] = model.cameras[image.camera_id].params.tolist()
            frame["world_to_camera"] = pose.tolist()

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
