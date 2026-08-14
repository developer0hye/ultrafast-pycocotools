"""The Rust JSON reader must produce exactly what ``json.load`` produces.

Not "equivalent" — equal, including float bit patterns. An annotation's
``area`` decides whether it counts as small, medium or large, so a value that
parses one ULP differently on either side of 32^2 would move AP_small. Both
parsers are correctly rounded, so this should hold by construction; the test is
here because "should hold by construction" is how bugs get shipped.
"""

from __future__ import annotations

import json
import math

import pytest
from conftest import REAL_GT

from ultrafast_pycocotools.coco import load_json


def assert_exactly_equal(a, b, path: str = "$") -> None:
    """Structural equality with float bit patterns compared, not values.

    ``==`` would call ``0.1 == 0.1`` True even if one were a different double,
    which is the whole thing we are testing, so floats go through
    ``float.hex``. It also distinguishes ``1`` from ``1.0``, which JSON does.
    """
    assert type(a) is type(b), f"{path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{path}: key order/set differs"
        for k in a:
            assert_exactly_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_exactly_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        if math.isnan(a):
            assert math.isnan(b), path
        else:
            assert a.hex() == b.hex(), f"{path}: {a!r} ({a.hex()}) vs {b!r} ({b.hex()})"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


LITERALS = [
    "{}",
    "[]",
    "null",
    "true",
    "[1, -1, 0, 9007199254740993]",
    '{"a": 1, "a": 2}',  # duplicate key: last wins, same as CPython
    '{"nested": {"deep": [[[1.5]]]}}',
    '"unicode: \\u00e9\\u4e2d\\ud83d\\ude00"',
    '[1e308, 1e-308, -0.0, 0.0, 1.7976931348623157e308, 5e-324]',
    '[0.1, 0.2, 0.30000000000000004, 177.71773017658128]',
    '[1.0, 1, 100.0, 1e2]',  # int vs float must not blur
    '{"empty_str": "", "esc": "a\\tb\\nc\\\\d\\"e"}',
]


@pytest.mark.parametrize("text", LITERALS)
def test_literals(text, tmp_path):
    p = tmp_path / "x.json"
    p.write_text(text, encoding="utf-8")
    assert_exactly_equal(json.loads(text), load_json(p))


def test_real_annotation_file():
    """The case that actually matters: a real COCO file, every float checked."""
    if not REAL_GT.exists():
        pytest.skip("COCO val2017 annotations not present")
    with open(REAL_GT) as f:
        want = json.load(f)
    assert_exactly_equal(want, load_json(REAL_GT))


def test_malformed_falls_back_to_stdlib_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        load_json(tmp_path / "nope.json")
