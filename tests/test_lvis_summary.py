"""Fused LVIS summary selection retains NumPy's exact reduction input."""
import numpy as np
import pytest

from ultrafast_pycocotools._lvis import valid_values


@pytest.mark.parametrize('precision', [True, False])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('categories', [[0, 1, 2, 3, 4], [4, 1, 1], []])
def test_summary_gather_preserves_logical_order_and_current_values(precision, strided, categories):
    shape = (4, 11, 5, 3, 2) if precision else (4, 5, 3, 2)
    array = np.random.default_rng(12).random(shape)
    array.ravel()[::7] = -1
    array.ravel()[::13] = np.nan
    array.ravel()[::19] = -0.0
    if strided:
        array = array[::-1, ..., ::-1]
    thresholds = [3, 0, 0]
    for replacement in (None, .125):
        if replacement is not None:
            array[...] = replacement
        if precision:
            selected = array[thresholds][:, :, categories, 1, 0]
        else:
            selected = array[thresholds][:, categories, 1, 0]
        expected = selected[selected > -1]
        actual = valid_values(array, thresholds, categories, 1, 0)
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        if expected.size:
            assert np.mean(actual).tobytes() == np.mean(expected).tobytes()


def test_summary_leaves_non_float64_and_subclass_indexing_to_numpy():
    assert valid_values(np.zeros((2, 3, 4, 1), np.float32), [0], [0], 0, 0) is None
    class Array(np.ndarray):
        pass
    assert valid_values(np.zeros((2, 3, 4, 1)).view(Array), [0], [0], 0, 0) is None
