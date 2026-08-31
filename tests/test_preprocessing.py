from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import pytest

from lumina.data.preprocessing import ViewConfig, extract_center_views


def test_center_view_export(tmp_path: Path) -> None:
    volume = np.arange(20 * 24 * 28, dtype=np.float32).reshape(20, 24, 28)
    nifti = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([1.0, 1.5, 2.0, 1.0])), nifti)
    outputs = extract_center_views(
        nifti,
        tmp_path / "views",
        stem="study",
        config=ViewConfig(output_size=64),
    )
    assert set(outputs) == {"axial", "coronal", "sagittal"}
    assert all(Image.open(path).size == (64, 64) for path in outputs.values())


def test_four_dimensional_input_requires_explicit_volume_index(tmp_path: Path) -> None:
    volume = np.zeros((8, 8, 8, 2), dtype=np.float32)
    nifti = tmp_path / "dynamic.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), nifti)
    with pytest.raises(ValueError, match="explicit volume_index"):
        extract_center_views(nifti, tmp_path / "views", stem="dynamic")
    outputs = extract_center_views(
        nifti,
        tmp_path / "views_indexed",
        stem="dynamic",
        config=ViewConfig(output_size=32, volume_index=1),
    )
    assert set(outputs) == {"axial", "coronal", "sagittal"}
