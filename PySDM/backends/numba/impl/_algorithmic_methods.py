import numba
import numpy as np

from PySDM.backends.numba.impl._atomic_operations import atomic_min

from PySDM.backends.numba import conf
from PySDM.backends.numba.storage import Storage


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def pair_indices(i, idx, is_first_in_pair):
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]
    return j, k


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell):
    omega = position_in_cell[dim, droplet]
    displacement[dim, droplet] = scheme(omega, courant[_l], courant[_r])


class AlgorithmicMethods:
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def adaptive_sdm_end_body(dt_left, n_cell, cell_start):
        end = 0
        for i in range(n_cell - 1, -1, -1):
            if dt_left[i] == 0:
                continue
            else:
                end = cell_start[i + 1]
                break
        return end

    @staticmethod
    def adaptive_sdm_end(dt_left, cell_start):
        return AlgorithmicMethods.adaptive_sdm_end_body(dt_left.data, len(dt_left), cell_start.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def adaptive_sdm_gamma_body(gamma, idx, length, n, cell_id, dt_left, dt, dt_range, is_first_in_pair,
                                stats_n_substep, stats_dt_min):
        dt_todo = np.empty(dt_left.shape, dtype=np.uint64)
        stats_dt_min_uint = np.empty(stats_dt_min.shape, dtype=np.uint64)
        coef = 2 ** 44
        for cid in numba.prange(len(dt_todo)):
            dt_todo[cid] = min(dt_left[cid], dt_range[1]) * coef
        stats_dt_min_uint[:] = stats_dt_min[:] * coef
        for i in range(length // 2):  # TODO #571
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = n[j] // n[k]
            dt_optimal = dt * prop / gamma[i]
            cid = cell_id[j]
            dt_optimal = max(dt_optimal, dt_range[0])
            atomic_min(dt_todo, cid, int(dt_optimal * coef))
            atomic_min(stats_dt_min_uint, cid, int(dt_optimal * coef))
        for i in numba.prange(length // 2):
            if gamma[i] == 0:
                continue
            j, _ = pair_indices(i, idx, is_first_in_pair)
            gamma[i] *= dt_todo[cell_id[j]] / dt / coef
        for cid in numba.prange(len(dt_todo)):
            dt_left[cid] -= dt_todo[cid] / coef
            if dt_todo[cid] > 0:
                stats_n_substep[cid] += 1
        stats_dt_min[:] = stats_dt_min_uint[:] / coef

    @staticmethod
    def adaptive_sdm_gamma(gamma, n, cell_id, dt_left, dt, dt_range, is_first_in_pair, stats_n_substep, stats_dt_min):
        return AlgorithmicMethods.adaptive_sdm_gamma_body(
            gamma.data, n.idx.data, len(n), n.data, cell_id.data,
            dt_left.data, dt, dt_range, is_first_in_pair.indicator.data, stats_n_substep.data, stats_dt_min.data)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    def calculate_displacement_body_1d(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in numba.prange(length):
            # Arakawa-C grid
            _l = cell_origin[0, droplet]
            _r = cell_origin[0, droplet] + 1
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    def calculate_displacement_body_2d(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in numba.prange(length):
            # Arakawa-C grid
            _l = (cell_origin[0, droplet], cell_origin[1, droplet])
            _r = (cell_origin[0, droplet] + 1 * (dim == 0), cell_origin[1, droplet] + 1 * (dim == 1))
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell)

    def calculate_displacement(self, dim, displacement, courant, cell_origin, position_in_cell):
        n_dims = len(courant.shape)
        scheme = self.formulae.particle_advection.displacement
        if n_dims == 1:
            AlgorithmicMethods.calculate_displacement_body_1d(dim, scheme, displacement.data, courant.data, cell_origin.data,
                                                              position_in_cell.data)
        elif n_dims == 2:
            AlgorithmicMethods.calculate_displacement_body_2d(dim, scheme, displacement.data, courant.data, cell_origin.data,
                                                              position_in_cell.data)
        else:
            raise NotImplementedError()

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # Note: in Numba 0.51 "np.dot() only supported on float and complex arrays"
    def cell_id_body(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin)

    @staticmethod
    def cell_id(cell_id, cell_origin, strides):
        return AlgorithmicMethods.cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def coalescence_body(n, idx, length, attributes, gamma, healthy, is_first_in_pair):
        for i in numba.prange(length // 2):
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)

            new_n = n[j] - gamma[i] * n[k]
            if new_n > 0:
                n[j] = new_n
                for a in range(0, len(attributes)):
                    attributes[a, k] += gamma[i] * attributes[a, j]
            else:  # new_n == 0
                n[j] = n[k] // 2
                n[k] = n[k] - n[j]
                for a in range(0, len(attributes)):
                    attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
                    attributes[a, k] = attributes[a, j]
            if n[k] == 0 or n[j] == 0:
                healthy[0] = 0

    @staticmethod
    def coalescence(n, idx, attributes, gamma, healthy, is_first_in_pair):
        AlgorithmicMethods.coalescence_body(n.data, idx.data, len(idx),
                                            attributes.data, gamma.data, healthy.data, is_first_in_pair.indicator.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def compute_gamma_body(gamma, rand, idx, length, n, cell_id,
                           collision_rate_deficit, collision_rate, is_first_in_pair):

        """
        return in "gamma" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)
        """
        for i in numba.prange(length // 2):
            gamma[i] = np.ceil(gamma[i] - rand[i])

            if gamma[i] == 0:
                continue

            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = n[j] // n[k]
            g = min(int(gamma[i]), prop)  # TODO #416: test asserting that min is not needed with adaptivity
            cid = cell_id[j]
            collision_rate[cid] += g * n[k]
            collision_rate_deficit[cid] += (int(gamma[i]) - g) * n[k]
            gamma[i] = g

    @staticmethod
    def compute_gamma(gamma, rand, n, cell_id,
                      collision_rate_deficit, collision_rate, is_first_in_pair):
        return AlgorithmicMethods.compute_gamma_body(
            gamma.data, rand.data, n.idx.data, len(n), n.data, cell_id.data,
            collision_rate_deficit.data, collision_rate.data, is_first_in_pair.indicator.data)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def flag_precipitated_body(cell_origin, position_in_cell, volume, n, idx, length, healthy):
        rainfall = 0.
        flag = len(idx)
        for i in range(length):
            if cell_origin[-1, idx[i]] + position_in_cell[-1, idx[i]] < 0:
                rainfall += volume[idx[i]] * n[idx[i]]
                idx[i] = flag
                healthy[0] = 0
        return rainfall

    @staticmethod
    def flag_precipitated(cell_origin, position_in_cell, volume, n, idx, length, healthy) -> float:
        return AlgorithmicMethods.flag_precipitated_body(
            cell_origin.data, position_in_cell.data, volume.data, n.data, idx.data, length, healthy.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def linear_collection_efficiency_body(params, output, radii, is_first_in_pair, idx, length, unit):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        output[:] = 0
        for i in numba.prange(length - 1):
            if is_first_in_pair[i]:
                if radii[idx[i]] > radii[idx[i + 1]]:
                    r = radii[idx[i]] / unit
                    r_s = radii[idx[i + 1]] / unit
                else:
                    r = radii[idx[i + 1]] / unit
                    r_s = radii[idx[i]] / unit
                p = r_s / r
                if p != 0 and p != 1:
                    G = (G1 / r) ** Mg + G2 + G3 * r
                    Gp = (1 - p) ** G
                    if Gp != 0:
                        D = D1 / r ** D2
                        E = E1 / r ** E2
                        F = (F1 / r) ** Mf + F2
                        output[i // 2] = A + B * p + D / p ** F + E / Gp
                        output[i // 2] = max(0, output[i // 2])

    @staticmethod
    def linear_collection_efficiency(params, output, radii, is_first_in_pair, unit):
        return AlgorithmicMethods.linear_collection_efficiency_body(
            params, output.data, radii.data, is_first_in_pair.indicator.data, radii.idx.data, len(is_first_in_pair), unit)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def interpolation_body(output, radius, factor, b, c):
        for i in numba.prange(len(radius)):
            r_id = int(factor * radius[i])
            r_rest = ((factor * radius[i]) % 1) / factor
            output[i] = b[r_id] + r_rest * c[r_id]

    @staticmethod
    def interpolation(output, radius, factor, b, c):
        return AlgorithmicMethods.interpolation_body(output.data, radius.data, factor, b.data, c.data)

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme="default"):
        class CellCaretaker:
            def __init__(self, idx, cell_start, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS['parallel']:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme == "counting_sort" or scheme == "counting_sort_parallel":
                    self.tmp_idx = Storage.empty(idx.shape, idx.dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty((numba.config.NUMBA_NUM_THREADS, len(cell_start)), dtype=int)

            def __call__(self, cell_id, cell_idx, cell_start, idx):
                length = len(idx)
                if self.scheme == "counting_sort":
                    AlgorithmicMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data, cell_idx.data, length, cell_start.data)
                elif self.scheme == "counting_sort_parallel":
                    AlgorithmicMethods._parallel_counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data, idx.data, cell_id.data, cell_idx.data, length, cell_start.data,
                        self.cell_starts.data)
                idx.data, self.tmp_idx.data = self.tmp_idx.data, idx.data

        return CellCaretaker(idx, cell_start, scheme)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def normalize_body(prob, cell_id, cell_idx, cell_start, norm_factor, dt, dv):
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = dt / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
        for d in numba.prange(prob.shape[0]):
            prob[d] *= norm_factor[cell_idx[cell_id[d]]]

    @staticmethod
    def normalize(prob, cell_id, cell_idx, cell_start, norm_factor, dt, dv):
        return AlgorithmicMethods.normalize_body(
            prob.data, cell_id.data, cell_idx.data, cell_start.data, norm_factor.data, dt, dv)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    def remove_zero_n_or_flagged(multiplicity, idx, length) -> int:
        flag = len(idx)
        new_length = length
        i = 0
        while i < new_length:
            if idx[i] == flag or multiplicity[idx[i]] == 0:
                new_length -= 1
                idx[i] = idx[new_length]
                idx[new_length] = flag
            else:
                i += 1
        return new_length

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def _counting_sort_by_cell_id_and_update_cell_start(new_idx, idx, cell_id, cell_idx, length, cell_start):
        cell_end = cell_start
        # Warning: Assuming len(cell_end) == n_cell+1
        cell_end[:] = 0
        for i in range(length):
            cell_end[cell_idx[cell_id[idx[i]]]] += 1
        for i in range(1, len(cell_end)):
            cell_end[i] += cell_end[i - 1]
        for i in range(length - 1, -1, -1):
            cell_end[cell_idx[cell_id[idx[i]]]] -= 1
            new_idx[cell_end[cell_idx[cell_id[idx[i]]]]] = idx[i]

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def _parallel_counting_sort_by_cell_id_and_update_cell_start(
            new_idx, idx, cell_id, cell_idx, length, cell_start, cell_start_p):
        cell_end_thread = cell_start_p
        # Warning: Assuming len(cell_end) == n_cell+1
        thread_num = cell_end_thread.shape[0]
        for t in numba.prange(thread_num):
            cell_end_thread[t, :] = 0
            for i in range(t * length // thread_num,
                           (t + 1) * length // thread_num if t < thread_num - 1 else length):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] += 1

        cell_start[:] = np.sum(cell_end_thread, axis=0)
        for i in range(1, len(cell_start)):
            cell_start[i] += cell_start[i - 1]

        tmp = cell_end_thread[0, :]
        tmp[:] = cell_end_thread[thread_num - 1, :]
        cell_end_thread[thread_num - 1, :] = cell_start[:]
        for t in range(thread_num - 2, -1, -1):
            cell_start[:] = cell_end_thread[t + 1, :] - tmp[:]
            tmp[:] = cell_end_thread[t, :]
            cell_end_thread[t, :] = cell_start[:]

        for t in numba.prange(thread_num):
            for i in range((t + 1) * length // thread_num - 1 if t < thread_num - 1 else length - 1,
                           t * length // thread_num - 1,
                           -1):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]
