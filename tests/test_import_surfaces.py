"""Verify the shared namespace remains lightweight in an installed checkout."""
from __future__ import annotations

import subprocess
import sys


def test_root_namespace_import_is_lightweight() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import ml_pipes; print('cv2' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_vlm_import_does_not_load_transformers_runtime() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from ml_pipes.vlm import TransformersModel; "
            "assert all(name not in sys.modules for name in ('torch', 'transformers'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
