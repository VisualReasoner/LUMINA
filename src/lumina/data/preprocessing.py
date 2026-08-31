from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ViewConfig:
    output_size: int = 512
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    volume_index: int | None = None


def _require_command(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"Required preprocessing command is not on PATH: {name}")
    return resolved


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def convert_dicom_series(dicom_dir: str | Path, output_dir: str | Path, *, stem: str = "volume") -> Path:
    source = Path(dicom_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _run([_require_command("dcm2niix"), "-z", "y", "-f", stem, "-o", str(destination), str(source)])
    candidates = sorted(destination.glob(f"{stem}*.nii.gz"))
    if not candidates:
        raise RuntimeError(f"dcm2niix produced no NIfTI file in {destination}")
    if len(candidates) != 1:
        names = [item.name for item in candidates]
        raise RuntimeError(
            "dcm2niix produced multiple NIfTI files for one requested study. "
            f"Select the intended series explicitly instead of using an arbitrary first file: {names}"
        )
    return candidates[0]


def preprocess_t1_mri(
    input_nifti: str | Path,
    output_dir: str | Path,
    *,
    mni_template: str | Path,
    prefix: str = "t1w",
) -> Path:
    """Apply N4 correction, SynthStrip extraction, and ANTs SyN normalization."""
    source = Path(input_nifti)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    n4 = destination / f"{prefix}_n4.nii.gz"
    stripped = destination / f"{prefix}_stripped.nii.gz"
    mask = destination / f"{prefix}_brain_mask.nii.gz"
    registration_prefix = destination / f"{prefix}_to_mni_"
    normalized = destination / f"{prefix}_mni.nii.gz"

    _run([_require_command("N4BiasFieldCorrection"), "-d", "3", "-i", str(source), "-o", str(n4)])
    _run([_require_command("mri_synthstrip"), "-i", str(n4), "-o", str(stripped), "-m", str(mask)])
    registration = shutil.which("antsRegistrationSyN.sh") or shutil.which("antsRegistrationSyNQuick.sh")
    if registration is None:
        raise RuntimeError("Required ANTs registration script is not on PATH.")
    _run(
        [
            registration,
            "-d",
            "3",
            "-f",
            str(mni_template),
            "-m",
            str(stripped),
            "-o",
            str(registration_prefix),
            "-t",
            "s",
        ]
    )
    warp = Path(f"{registration_prefix}1Warp.nii.gz")
    affine = Path(f"{registration_prefix}0GenericAffine.mat")
    _run(
        [
            _require_command("antsApplyTransforms"),
            "-d",
            "3",
            "-i",
            str(stripped),
            "-r",
            str(mni_template),
            "-o",
            str(normalized),
            "-t",
            str(warp),
            "-t",
            str(affine),
        ]
    )
    return normalized


def _scale_uint8(array: np.ndarray, config: ViewConfig) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [config.lower_percentile, config.upper_percentile])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    clipped = np.clip(array, low, high)
    scaled = (clipped - low) / (high - low)
    return np.round(scaled * 255.0).astype(np.uint8)


def _physical_resize(array: np.ndarray, spacing: tuple[float, float], output_size: int) -> Image.Image:
    image = Image.fromarray(array, mode="L")
    physical_width = max(1e-6, array.shape[1] * spacing[1])
    physical_height = max(1e-6, array.shape[0] * spacing[0])
    scale = min(output_size / physical_width, output_size / physical_height)
    width = max(1, min(output_size, round(physical_width * scale)))
    height = max(1, min(output_size, round(physical_height * scale)))
    resized = image.resize((width, height), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("L", (output_size, output_size), color=0)
    canvas.paste(resized, ((output_size - width) // 2, (output_size - height) // 2))
    return canvas


def extract_center_views(
    input_nifti: str | Path,
    output_dir: str | Path,
    *,
    stem: str,
    config: ViewConfig | None = None,
) -> dict[str, Path]:
    cfg = config or ViewConfig()
    image = nib.as_closest_canonical(nib.load(str(input_nifti)))
    volume = np.asarray(image.get_fdata(dtype=np.float32))
    if volume.ndim == 4:
        if cfg.volume_index is None:
            raise ValueError(
                f"A 4D volume requires an explicit volume_index; found shape {volume.shape}."
            )
        if not 0 <= cfg.volume_index < volume.shape[3]:
            raise ValueError(f"volume_index {cfg.volume_index} is outside 0..{volume.shape[3] - 1}.")
        volume = volume[..., cfg.volume_index]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D or explicitly indexed 4D NIfTI volume, found shape {volume.shape}.")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    scaled = _scale_uint8(volume, cfg)
    slices = {
        "sagittal": (scaled[scaled.shape[0] // 2, :, :].T, (spacing[2], spacing[1])),
        "coronal": (scaled[:, scaled.shape[1] // 2, :].T, (spacing[2], spacing[0])),
        "axial": (scaled[:, :, scaled.shape[2] // 2].T, (spacing[1], spacing[0])),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for view, (array, plane_spacing) in slices.items():
        oriented = np.flipud(array)
        output = destination / f"{stem}_{view}.png"
        _physical_resize(oriented, plane_spacing, cfg.output_size).save(output)
        outputs[view] = output
    return outputs
