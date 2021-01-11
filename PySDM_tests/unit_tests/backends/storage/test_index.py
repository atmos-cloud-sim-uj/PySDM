"""
Created at 05.10.2020
"""

import numpy as np

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


class TestIndex:

    @staticmethod
    def test_remove_zeros(backend):
        # Arrange
        n_sd = 44
        idx = backend.Index.empty(n_sd)
        data = np.ones(n_sd).astype(np.int64)
        data[0], data[n_sd // 2], data[-1] = 0, 0, 0
        data = backend.Storage.from_ndarray(data)
        data = backend.IndexedStorage.indexed(storage=data, idx=idx)

        # Act
        idx.remove_zeros(data)

        # Assert
        assert len(idx) == n_sd - 3
        assert (backend.Storage.to_ndarray(data)[idx.to_ndarray()[:len(idx)]] > 0).all()
