"""
Koehler-curve equilibrium in unsaturated conditions
"""
from ..backends.numba.toms748 import toms748_solve
from ..physics import constants as const
from ..backends.numba.conf import JIT_FLAGS
from numba import prange, njit
import numpy as np

default_rtol = 1e-5
default_max_iters = 64


def r_wet_init(r_dry: np.ndarray, environment, kappa, cell_id: np.ndarray = None,
               rtol=default_rtol, max_iters=default_max_iters):
    if cell_id is None:
        cell_id = np.zeros_like(r_dry, dtype=int)

    T = environment["T"].to_ndarray()
    RH = environment["RH"].to_ndarray()

    formulae = environment.core.formulae
    r_cr = formulae.hygroscopicity.r_cr
    RH_eq = formulae.hygroscopicity.RH_eq
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    within_tolerance = formulae.trivia.within_tolerance

    jit_flags = {**JIT_FLAGS, **{'fastmath': formulae.fastmath, 'cache': False}}

    @njit(**{**jit_flags, 'parallel': False})
    def minfun(r, T, RH, kp, rd3):
        sgm = sigma(T, v_wet=phys_volume(radius=r), v_dry=const.pi_4_3 * rd3)
        return RH - RH_eq(r, T, kp, rd3, sgm)

    @njit(**jit_flags)
    def r_wet_init_impl(r_dry: np.ndarray, iters, T, RH, cell_id: np.ndarray, kappa, rtol,
                        RH_range=(0, 1)):
        r_wet = np.empty_like(r_dry)
        for i in prange(len(r_dry)):
            r_d = r_dry[i]
            cid = cell_id[i]
            # root-finding initial guess
            a = r_d
            b = r_cr(kappa, r_d**3, T[cid], const.sgm_w)
            # minimisation
            args = (
                T[cid],
                np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid])),
                kappa,
                r_d**3
            )
            fa = minfun(a, *args)
            fb = minfun(b, *args)
            r_wet[i], iters[i] = toms748_solve(minfun, args, a, b, fa, fb, rtol=rtol, max_iter=max_iters,
                                                 within_tolerance=within_tolerance)
        return r_wet

    iters = np.empty_like(r_dry, dtype=int)
    r_wet = r_wet_init_impl(r_dry, iters, T, RH, cell_id, kappa, rtol)
    assert (iters != max_iters).all()
    return r_wet


