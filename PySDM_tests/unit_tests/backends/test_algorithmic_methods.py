import os

import numpy as np
import pytest

from PySDM.backends.numba.impl._algorithmic_methods import pair_indices
from PySDM.storages.index import make_Index
from PySDM.storages.indexed_storage import make_IndexedStorage
from PySDM.storages.pair_indicator import make_PairIndicator
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


@pytest.mark.parametrize("i, idx, is_first_in_pair, expected", [
    (0, (0, 1), (True, False), (0, 1)),
    (0, (1, 0), (True, False), (1, 0)),
    (0, (0, 1, 2), (False, True), (1, 2)),
])
def test_pair_indices(i, idx, is_first_in_pair, expected):
    # Arrange
    sut = pair_indices if 'NUMBA_DISABLE_JIT' in os.environ else pair_indices.py_func

    # Act
    actual = sut(i, idx, is_first_in_pair)

    # Assert
    assert expected == actual


N = 2*17 + 2

class TestAlgorithmicMethods:
    @staticmethod
    @pytest.mark.parametrize("dt_left, cell_start, expected", [
        ((4, 5, 4.5, 0, 0), (0, 1, 2, 3, 4, 5), 3),
        ((4, 5, 4.5, 3, .1), (0, 1, 2, 3, 4, 5), 5)
    ])
    def test_adaptive_sdm_end(backend, dt_left, cell_start, expected):
        # Arrange
        dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        cell_start = backend.Storage.from_ndarray(np.asarray(cell_start))

        # Act
        actual = backend.adaptive_sdm_end(dt_left, cell_start)

        # Assert
        assert actual == expected

    @staticmethod
    @pytest.mark.parametrize(
        "gamma, idx, n, cell_id, dt_left, dt, dt_max, is_first_in_pair, expected_dt_left, expected_n_substep", [
            ((10.,), (0, 1), (44, 44), (0, 0), (10.,), 10., 10., (True, False), (9.,), (1,)),
            ([1.]*(N//2-1) + [10.] + [1.]*(N//2-1), range(N), [44]*N, [0]*N, (10.,), 10., 10., [i%2 == 0 for i in range(N)], (9.,), (1,)),
            ((10.,), (0, 1), (44, 44), (0, 0), (10.,), 10., .1, (True, False), (9.9,), (1,)),
            ((0.,), (0, 1), (44, 44), (0, 0), (10.,), 10., 10., (False, True), (0.,), (1,)),
            ((10.,), (0, 1), (440, 44), (0, 0), (10.,), 10., 10., (True, False), (0.,), (1,)),
            ((.5, 6), (0, 1, 2, 3, 4), (44, 44, 22, 33, 11), (0, 0, 0, 1, 1), (10., 10), 10., 10., (True, False, False, True, False), (0., 5.), (1, 1)),
        ])
    def test_adaptive_sdm_gamma(backend, gamma, idx, n, cell_id, dt_left, dt, dt_max, is_first_in_pair, expected_dt_left, expected_n_substep):
        # Arrange
        _gamma = backend.Storage.from_ndarray(np.asarray(gamma))
        _idx = make_Index(backend).from_ndarray(np.asarray(idx))
        _n = make_IndexedStorage(backend).from_ndarray(_idx, np.asarray(n))
        _cell_id = backend.Storage.from_ndarray(np.asarray(cell_id))
        _dt_left = backend.Storage.from_ndarray(np.asarray(dt_left))
        _is_first_in_pair = make_PairIndicator(backend)(len(n))
        _is_first_in_pair.indicator[:] = np.asarray(is_first_in_pair)
        _n_substep = backend.Storage.from_ndarray(np.zeros_like(dt_left, dtype=int))
        _dt_min = backend.Storage.from_ndarray(np.zeros_like(dt_left))
        dt_range = (np.nan, dt_max)

        # Act
        backend.adaptive_sdm_gamma(_gamma, _n, _cell_id, _dt_left, dt, dt_range, _is_first_in_pair, _n_substep, _dt_min)

        # Assert
        np.testing.assert_array_almost_equal(_dt_left.to_ndarray(), np.asarray(expected_dt_left))
        expected_gamma = np.empty_like(np.asarray(gamma))
        for i in range(len(idx)):
            if is_first_in_pair[i]:
                expected_gamma[i // 2] = (dt - np.asarray(expected_dt_left[cell_id[i]])) / dt * np.asarray(gamma)[
                    i // 2]
        np.testing.assert_array_almost_equal(_gamma.to_ndarray(), expected_gamma)
        np.testing.assert_array_equal(_n_substep, np.asarray(expected_n_substep))
