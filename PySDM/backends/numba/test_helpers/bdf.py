"""
Created at 09.01.2020
"""

import PySDM.physics.constants as const
from PySDM.physics import formulae as phys
from PySDM.backends.numba.numba import Numba
from PySDM.backends.numba.conf import JIT_FLAGS
import numpy as np
import numba
import scipy.integrate
import types
import warnings

idx_thd = 0
idx_x = 1
rtol = 1e-4


def patch_core(core):
    core.condensation_solver = _make_solve(core.backend.formulae)
    core.condensation = types.MethodType(_bdf_condensation, core)


def _bdf_condensation(core, kappa, rtol_x, rtol_thd, counters, RH_max, success, cell_order):
    n_threads = 1
    if core.particles.has_attribute("temperature"):
        raise NotImplementedError()

    func = Numba._condensation
    if not numba.config.DISABLE_JIT:
        func = func.py_func
    func(
        solver=core.condensation_solver,
        n_threads=n_threads,
        n_cell=core.mesh.n_cell,
        cell_start_arg=core.particles.cell_start.data,
        v=core.particles["volume"].data,
        particle_temperatures=np.empty(0),
        v_cr=None,
        n=core.particles['n'].data,
        vdry=core.particles["dry volume"].data,
        idx=core.particles._Particles__idx.data,
        rhod=core.env["rhod"].data,
        thd=core.env["thd"].data,
        qv=core.env["qv"].data,
        dv_mean=core.env.dv,
        prhod=core.env.get_predicted("rhod").data,
        pthd=core.env.get_predicted("thd").data,
        pqv=core.env.get_predicted("qv").data,
        kappa=kappa,
        rtol_x=rtol_x,
        rtol_thd=rtol_thd,
        dt=core.dt,
        counter_n_substeps=counters['n_substeps'],
        counter_n_activating=counters['n_activating'],
        counter_n_deactivating=counters['n_deactivating'],
        counter_n_ripening=counters['n_ripening'],
        cell_order=cell_order,
        RH_max=RH_max.data,
        success=success.data
    )


def _make_solve(formulae):
    x = formulae.condensation_coordinate.x
    volume = formulae.condensation_coordinate.volume
    dx_dt = formulae.condensation_coordinate.dx_dt
    pvs_C = formulae.saturation_vapour_pressure.pvs_Celsius
    lv = formulae.latent_heat.lv

    @numba.njit(**{**JIT_FLAGS, **{'parallel': False, 'inline': 'always'}})
    def _ql(n, x, m_d_mean):
        return np.sum(n * volume(x)) * const.rho_w / m_d_mean

    @numba.njit(**JIT_FLAGS)
    def _impl(dy_dt, x, T, p, n, RH, kappa, rd, thd, dot_thd, dot_qv, m_d_mean, rhod_mean, pvs, lv):
        for i in numba.prange(len(x)):
            dy_dt[idx_x + i] = dx_dt(x[i], phys.dr_dt_MM(phys.radius(volume(x[i])), T, p, RH, lv, pvs, kappa, rd[i]))
        dqv_dt = dot_qv - np.sum(n * volume(x) * dy_dt[idx_x:]) * const.rho_w / m_d_mean
        dy_dt[idx_thd] = dot_thd + phys.dthd_dt(rhod_mean, thd, T, dqv_dt, lv)

    @numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
    def _odesys(t, y, kappa, dry_volume, n, dthd_dt, dqv_dt, m_d_mean, rhod_mean, qt):
        rd = phys.radius(volume=dry_volume)

        thd = y[idx_thd]
        x = y[idx_x:]

        qv = qt + dqv_dt * t - _ql(n, x, m_d_mean)
        T, p, pv = phys.temperature_pressure_pv(rhod_mean, thd, qv)
        pvs = pvs_C(T - const.T0)
        RH = pv / pvs

        dy_dt = np.empty_like(y)
        _impl(dy_dt, x, T, p, n, RH, kappa, rd, thd, dthd_dt, dqv_dt, m_d_mean, rhod_mean, pvs, lv(T))
        return dy_dt

    def solve(
            v, particle_temperatures, v_cr, n, vdry,
            cell_idx, kappa, thd, qv,
            dthd_dt, dqv_dt, m_d_mean, rhod_mean,
            rtol_x, rtol_thd, dt, substeps
    ):
        n_sd_in_cell = len(cell_idx)
        y0 = np.empty(n_sd_in_cell + idx_x)
        y0[idx_thd] = thd
        y0[idx_x:] = x(v[cell_idx])
        qt = qv + _ql(n[cell_idx], y0[idx_x:], m_d_mean)
        args = (kappa, vdry[cell_idx], n[cell_idx], dthd_dt, dqv_dt, m_d_mean, rhod_mean, qt)
        if dthd_dt == 0 and dqv_dt == 0 and (_odesys(0, y0, *args)[idx_x] == 0).all():
            y1 = y0
        else:
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("ignore")
                integ = scipy.integrate.solve_ivp(
                    fun=_odesys,
                    args=args,
                    t_span=(0, dt),
                    t_eval=(dt,),
                    y0=y0,
                    rtol=rtol,
                    atol=0,
                    method="BDF"
                )
            assert integ.success, integ.message
            y1 = integ.y[:, 0]

        m_new = 0
        for i in range(n_sd_in_cell):
            v_new = volume(y1[idx_x + i])
            m_new += n[cell_idx[i]] * v_new * const.rho_w
            v[cell_idx[i]] = v_new

        return integ.success, qt - m_new / m_d_mean, y1[idx_thd], 1, 1, 1, 1, np.nan

    return solve