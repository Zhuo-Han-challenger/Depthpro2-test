"""Render a 3DGS preview alongside the input RGB image and metric depth."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from matplotlib.colors import LogNorm
from plyfile import PlyData

from rgb_images import read_rgb


def read_splats(path):
    vertices = PlyData.read(path)["vertex"].data
    if not len(vertices):
        raise ValueError("the PLY contains no splats")

    def columns(fields):
        return np.column_stack([vertices[field] for field in fields]).astype(np.float64)

    points = columns(("x", "y", "z"))
    sh_color = columns(("f_dc_0", "f_dc_1", "f_dc_2"))
    colors = np.clip(sh_color / np.sqrt(4 * np.pi) + 0.5, 0, 1)
    scales = np.exp(columns(("scale_0", "scale_1", "scale_2")))
    opacity = 1 / (1 + np.exp(-np.clip(vertices["opacity"].astype(np.float64), -80, 80)))

    # The exporter shares one orientation across all splats from each view.
    quats = columns(("rot_1", "rot_2", "rot_3", "rot_0"))
    unique, indices = np.unique(quats, axis=0, return_inverse=True)
    rotations = np.array([pycolmap.Rotation3d(quat).matrix() for quat in unique])
    world_axes = rotations[indices] * scales[:, None, :]
    return points, colors, opacity, world_axes


def render(splats, view, intrinsics, width, height):
    points, colors, opacity, world_axes = splats
    rotation = view[:, :3]
    camera_points = points @ rotation.T + view[:, 3]
    visible = camera_points[:, 2] > 0.01
    x, y, z = camera_points[visible].T
    colors = colors[visible]
    opacity = opacity[visible]
    fx, fy, cx, cy = intrinsics
    u, v = fx * x / z + cx - 0.5, fy * y / z + cy - 0.5

    # Approximate each projected splat by a 2D Gaussian at its center.
    jacobian = np.zeros((len(z), 2, 3))
    jacobian[:, 0, 0], jacobian[:, 1, 1] = fx / z, fy / z
    jacobian[:, 0, 2], jacobian[:, 1, 2] = -fx * x / z**2, -fy * y / z**2
    projected = jacobian @ rotation @ world_axes[visible]
    covariance = projected @ projected.transpose(0, 2, 1) + 0.3 * np.eye(2)
    precision = np.linalg.inv(covariance)
    radius_x = 3 * np.sqrt(covariance[:, 0, 0])
    radius_y = 3 * np.sqrt(covariance[:, 1, 1])

    output = np.zeros((height, width, 3), dtype=np.float32)
    remaining = np.ones((height, width), dtype=np.float32)
    for index in np.argsort(z):
        left = max(0, int(np.ceil(u[index] - radius_x[index])))
        right = min(width, int(np.floor(u[index] + radius_x[index])) + 1)
        top = max(0, int(np.ceil(v[index] - radius_y[index])))
        bottom = min(height, int(np.floor(v[index] + radius_y[index])) + 1)
        if left >= right or top >= bottom:
            continue
        dx = np.arange(left, right) - u[index]
        dy = np.arange(top, bottom)[:, None] - v[index]
        inverse = precision[index]
        distance = inverse[0, 0] * dx**2 + 2 * inverse[0, 1] * dx * dy + inverse[1, 1] * dy**2
        alpha = np.where(distance <= 9, opacity[index] * np.exp(-0.5 * distance), 0)
        region = np.s_[top:bottom, left:right]
        output[region] += (remaining[region] * alpha)[..., None] * colors[index]
        remaining[region] *= 1 - alpha
    return output.clip(0, 1)


def preview(manifest, case_dir, output, shift):
    model_dir = case_dir / "gen_video" / "sparse" / "0"
    reconstruction = pycolmap.Reconstruction(model_dir)
    frame = json.loads(manifest.read_text(encoding="utf-8"))["frames"][0]
    source_path = manifest.parent / frame["image"]
    name = frame.get("name", source_path.name)
    image = reconstruction.find_image_with_name(name)
    if image is None:
        raise ValueError(f"camera for {name!r} is missing from images.bin")
    camera = reconstruction.cameras[image.camera_id]
    if camera.model_name != "PINHOLE":
        raise ValueError("the preview expects the undistorted PINHOLE camera exported by this repo")

    original = read_rgb(source_path)
    with np.load(manifest.parent / frame["depth"], allow_pickle=False) as prediction:
        depth = prediction["depth"]
    if original.size != (camera.width, camera.height) or depth.shape != original.size[::-1]:
        raise ValueError("RGB image, depth map and COLMAP camera dimensions must match")
    valid_depth = np.isfinite(depth) & (depth > 0)
    if not valid_depth.any():
        raise ValueError("the selected image has no finite positive depth")

    width = min(camera.width, 768)
    height = max(1, round(camera.height * width / camera.width))
    sx, sy = width / camera.width, height / camera.height
    intrinsics = camera.params * [sx, sy, sx, sy]
    view = image.cam_from_world().matrix()
    shifted = view.copy()
    # Moving the camera along its local right axis subtracts x from camera-space points.
    shifted[0, 3] -= shift
    splats = read_splats(model_dir / "3dgs.ply")
    source_render = render(splats, view, intrinsics, width, height)
    shifted_render = render(splats, shifted, intrinsics, width, height)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), layout="constrained")
    near, far = np.percentile(depth[valid_depth], [2, 98])
    far = max(far, near * 1.01)
    cmap = plt.colormaps["viridis"].with_extremes(bad="#181b20")
    axes[0, 0].imshow(original)
    mapped = axes[0, 1].imshow(
        np.ma.masked_where(~valid_depth, depth), cmap=cmap, norm=LogNorm(near, far)
    )
    axes[1, 0].imshow(source_render)
    axes[1, 1].imshow(shifted_render)
    titles = (
        "Original RGB",
        f"Dense depth · {camera.width} × {camera.height}",
        "3DGS · original camera",
        f"3DGS · camera shifted {abs(shift):g} m {'right' if shift >= 0 else 'left'}",
    )
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_axis_off()
    bar = fig.colorbar(mapped, ax=axes[0, 1], orientation="horizontal", pad=0.05, extend="both")
    bar.set_label("Depth (m, log scale) · dark gray: invalid or infinite depth")
    bar.ax.minorticks_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, format="png")
    plt.close(fig)
    print(f"Wrote {output} ({len(splats[0]):,} splats)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="frame manifest used by export-colmap.py")
    parser.add_argument("case_dir", type=Path, help="case directory from the COLMAP/Gaussian export")
    parser.add_argument("output", type=Path, help="output PNG path")
    parser.add_argument("--shift-m", type=float, default=0.65, help="camera-right shift in meters")
    args = parser.parse_args()
    if not np.isfinite(args.shift_m):
        parser.error("--shift-m must be finite")
    try:
        preview(args.manifest, args.case_dir, args.output, args.shift_m)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        parser.exit(1, f"Preview failed: {exc}\n")


if __name__ == "__main__":
    main()
