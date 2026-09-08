"""Published-input verification must reject changed inputs before scoring."""
import os
from pathlib import Path
import subprocess
import sys


def test_published_input_mismatch_stops_before_evaluators(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'changed-seed'
    result = subprocess.run(
        [sys.executable, str(root / 'bench/reproduce.py'), 'quick', '--out', str(out),
         '--seed', '99', '--verify-published', 'quick'],
        capture_output=True, text=True, env={**os.environ, 'PYTHONUTF8': '1'},
    )
    assert result.returncode != 0
    assert 'Published input mismatch' in result.stderr
    assert (out / 'generation.log').is_file()
    assert not (out / 'pycocotools').exists()
    assert not (out / 'ultrafast').exists()
