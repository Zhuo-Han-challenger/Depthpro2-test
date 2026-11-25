"""Shared RGB loading for the COLMAP and Gaussian workflows."""

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

if register_heif_opener is not None:
    register_heif_opener()


def supported_image_suffixes():
    extensions = Image.registered_extensions()
    non_photos = {"BUFR", "EPS", "FITS", "GRIB", "HDF5", "MPEG", "WMF"}
    return {
        suffix.lower() for suffix, fmt in extensions.items()
        if fmt in Image.OPEN and fmt not in non_photos and suffix.lower() != ".pfm"
    }


def read_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")
