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
