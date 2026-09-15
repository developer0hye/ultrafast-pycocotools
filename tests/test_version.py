"""``__version__`` must be the version the wheel was stamped with, which is the Cargo workspace version."""
from importlib.metadata import version as distribution_version
from pathlib import Path
import re

import ultrafast_pycocotools


def test_version_matches_installed_distribution():
    assert ultrafast_pycocotools.__version__ == distribution_version("ultrafast-pycocotools")


def test_version_matches_cargo_workspace():
    cargo = Path(__file__).resolve().parents[1] / "Cargo.toml"
    match = re.search(r'^\[workspace\.package\](?:.*\n)*?version = "([^"]+)"', cargo.read_text(), re.MULTILINE)
    assert match is not None, "workspace.package.version not found in Cargo.toml"
    assert ultrafast_pycocotools.__version__ == match.group(1), (
        "installed package is stale relative to Cargo.toml; reinstall with `pip install .`"
    )
