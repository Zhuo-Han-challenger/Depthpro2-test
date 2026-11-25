# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=11", "pillow-heif", "pycolmap>=4.2,<5"]
# ///
"""Export Depth Pro 2 predictions as a colored COLMAP reconstruction."""

import argparse
import json
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

import numpy as np
import pycolmap

from rgb_images import read_rgb


GAUSSIAN_PROPERTIES = (
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)
SH_DC_CONSTANT = 1 / np.sqrt(4 * np.pi)


def write_gaussian_vertices(stream, points, colors, depth, focal_lengths, stride, camera_to_world):
    vertices = np.zeros((len(points), len(GAUSSIAN_PROPERTIES)), dtype="<f4")
    vertices[:, :3] = points
    vertices[:, 6:9] = (colors / 255.0 - 0.5) / SH_DC_CONSTANT
    vertices[:, 9] = np.log(9.0)  # Logit of the initial opacity, 0.9.

    # Half the spacing between depth samples, with a thinner camera-space z axis.
    log_footprint = np.log(depth[:, None]) + np.log(stride / 2) - np.log(focal_lengths)
    vertices[:, 10:12] = log_footprint
    vertices[:, 12] = log_footprint.min(axis=1) - np.log(4.0)

    # PyCOLMAP quaternions are xyzw. Gaussian PLY stores wxyz in world axes.
    qx, qy, qz, qw = camera_to_world.rotation.quat
    vertices[:, 13:] = (qw, qx, qy, qz)
    stream.write(vertices.tobytes())


def write_gaussian_ply(path, body, count):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property float {name}" for name in GAUSSIAN_PROPERTIES)
    header.append("end_header\n")
    body.seek(0)
    with path.open("wb") as output:
        output.write("\n".join(header).encode("ascii"))
        shutil.copyfileobj(body, output)


def load_calibration(frame, prediction, width, height, single_view):
    intrinsics = frame.get("intrinsics")
    if intrinsics is None:
        if not single_view:
            raise ValueError("multi-view export requires intrinsics for every frame")
        focal = float(prediction["focallength_px"].item())
        intrinsics = [focal, focal, width / 2, height / 2]
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if (
        intrinsics.shape != (4,)
        or not np.isfinite(intrinsics).all()
        or np.any(intrinsics[:2] <= 0)
    ):
        raise ValueError("intrinsics must be [fx, fy, cx, cy] with positive focal lengths")

    matrix = frame.get("world_to_camera")
    if matrix is None:
        if not single_view:
            raise ValueError("multi-view export requires world_to_camera for every frame")
        matrix = np.eye(4)[:3]
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4) and np.allclose(matrix[3], [0, 0, 0, 1]):
        matrix = matrix[:3]
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("world_to_camera must be a finite 3×4 or homogeneous 4×4 matrix")
    rotation = matrix[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("world_to_camera must contain a proper rotation")
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), matrix[:, 3])
    return intrinsics, pose


def add_frame(
    reconstruction, frame, manifest_dir, frame_id, single_view, stride, max_depth, gaussian_body
):
    image_path = manifest_dir / frame["image"]
    name = frame.get("name", image_path.name)
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or any(character in name for character in ("\\", "\x00", "\r", "\n"))
    ):
        raise ValueError("image name must be a relative COLMAP image path")

    with read_rgb(image_path) as source:
        rgb = np.asarray(source)
    height, width = rgb.shape[:2]

    with np.load(manifest_dir / frame["depth"], allow_pickle=False) as prediction:
        depth = prediction["depth"]
        if depth.shape != (height, width) or depth.dtype.kind != "f":
            raise ValueError(f"{name}: depth must be floating-point with shape {(height, width)}")
        intrinsics, pose = load_calibration(frame, prediction, width, height, single_view)

    # Mask before projection so sky and invalid samples never become 3D points.
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0) & (sampled <= max_depth)
    rows, cols = np.nonzero(valid)
    z = sampled[rows, cols].astype(np.float64)
    rows *= stride
    cols *= stride

    # COLMAP's top-left pixel center is at (0.5, 0.5).
    keypoints = np.column_stack((cols + 0.5, rows + 0.5))
    fx, fy, cx, cy = intrinsics
    x, y = keypoints.T
    camera_points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    camera_to_world = pose.inverse()
    world_points = camera_to_world * camera_points
    if not np.isfinite(world_points).all() or np.any(
        np.abs(world_points) > np.finfo(np.float32).max
    ):
        raise ValueError(f"{name}: projected points exceed the PLY coordinate range")

    camera = pycolmap.Camera(
        model="PINHOLE", camera_id=frame_id, width=width, height=height, params=intrinsics
    )
    reconstruction.add_camera_with_trivial_rig(camera)
    image = pycolmap.Image(name=name, keypoints=keypoints, camera_id=frame_id, image_id=frame_id)
    reconstruction.add_image_with_trivial_frame(image, pose)

    colors = rgb[rows, cols]
    for index, (point, color) in enumerate(zip(world_points, colors, strict=True)):
        track = pycolmap.Track()
        track.add_element(frame_id, index)
        reconstruction.add_point3D(point, track, color)
    if gaussian_body is not None:
        write_gaussian_vertices(
            gaussian_body, world_points, colors, z, intrinsics[:2], stride, camera_to_world
        )
    return name


def export(manifest, case_dir, stride, max_depth, gaussian_ply=False):
    frames = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest must contain a nonempty frames list")
    output_dir = case_dir / "gen_video" / "sparse" / "0"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"export destination already exists: {output_dir}")

    reconstruction = pycolmap.Reconstruction()
    with (tempfile.TemporaryFile() if gaussian_ply else nullcontext()) as gaussian_body:
        names = set()
        for frame_id, frame in enumerate(frames, start=1):
            name = add_frame(
                reconstruction, frame, manifest.parent, frame_id, len(frames) == 1,
                stride, max_depth, gaussian_body,
            )
            if name in names:
                raise ValueError(f"duplicate COLMAP image name: {name}")
            names.add(name)
        point_count = reconstruction.num_points3D()
        if not point_count:
            raise ValueError("no sampled pixels have valid depth")
        if not reconstruction.is_valid():
            raise ValueError("generated COLMAP reconstruction is inconsistent")

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
            staged = Path(tmp) / "model"
            staged.mkdir()
            reconstruction.write_binary(staged)
            reconstruction.export_PLY(staged / "points3D.ply")
            if gaussian_body is not None:
                write_gaussian_ply(staged / "3dgs.ply", gaussian_body, point_count)
            staged.rename(output_dir)
    return output_dir, point_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest of RGB images and depth NPZs")
    parser.add_argument("case_dir", type=Path, help="write under CASE_DIR/gen_video/sparse/0")
    parser.add_argument("--stride", type=int, default=8, help="pixel spacing for point sampling")
    parser.add_argument("--max-depth", type=float, default=np.inf, help="maximum depth in meters")
    parser.add_argument("--gaussian-ply", action="store_true", help="also write 3dgs.ply")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if np.isnan(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    try:
        output_dir, point_count = export(
            args.manifest, args.case_dir, args.stride, args.max_depth, args.gaussian_ply
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {point_count:,} points to {output_dir}")
    if args.gaussian_ply:
        print(f"Wrote {point_count:,} Gaussian splats to {output_dir / '3dgs.ply'}")


if __name__ == "__main__":
    main()
