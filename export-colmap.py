# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=11", "pycolmap>=4.2,<5"]
# ///
"""Export Depth Pro 2 predictions as a colored COLMAP reconstruction."""

import argparse
import json
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pycolmap
from PIL import Image, ImageOps


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


def add_frame(reconstruction, frame, manifest_dir, frame_id, single_view, stride, max_depth):
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

    with Image.open(image_path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
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
    world_points = pose.inverse() * camera_points
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

    for index, (point, color) in enumerate(zip(world_points, rgb[rows, cols], strict=True)):
        track = pycolmap.Track()
        track.add_element(frame_id, index)
        reconstruction.add_point3D(point, track, color)
    return name


def export(manifest, case_dir, stride, max_depth):
    frames = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest must contain a nonempty frames list")
    output_dir = case_dir / "gen_video" / "sparse" / "0"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"export destination already exists: {output_dir}")

    reconstruction = pycolmap.Reconstruction()
    names = set()
    for frame_id, frame in enumerate(frames, start=1):
        name = add_frame(
            reconstruction, frame, manifest.parent, frame_id, len(frames) == 1, stride, max_depth
        )
        if name in names:
            raise ValueError(f"duplicate COLMAP image name: {name}")
        names.add(name)
    if not reconstruction.num_points3D():
        raise ValueError("no sampled pixels have valid depth")
    if not reconstruction.is_valid():
        raise ValueError("generated COLMAP reconstruction is inconsistent")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "model"
        staged.mkdir()
        reconstruction.write_binary(staged)
        reconstruction.export_PLY(staged / "points3D.ply")
        staged.rename(output_dir)
    return output_dir, reconstruction.num_points3D()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest of RGB images and depth NPZs")
    parser.add_argument("case_dir", type=Path, help="write under CASE_DIR/gen_video/sparse/0")
    parser.add_argument("--stride", type=int, default=8, help="pixel spacing for point sampling")
    parser.add_argument("--max-depth", type=float, default=np.inf, help="maximum depth in meters")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if np.isnan(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    try:
        output_dir, point_count = export(args.manifest, args.case_dir, args.stride, args.max_depth)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {point_count:,} points to {output_dir}")


if __name__ == "__main__":
    main()
