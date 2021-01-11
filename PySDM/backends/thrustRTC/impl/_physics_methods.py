"""
Created at 20.03.2020
"""

from ..conf import trtc
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
import PySDM.physics.constants as const
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver


class PhysicsMethods:
    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def explicit_in_space(omega, c_l, c_r):
        return "c_l * (1 - omega) + c_r * omega;"

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def implicit_in_space(omega, c_l, c_r):
        """
        see eqs 14-16 in Arabas et al. 2015 (libcloudph++)
        """
        result = "(omega * (c_r - c_l) + c_l) / (1 - (c_r - c_l));"
        return result

    __temperature_pressure_RH_body = trtc.For(["rhod", "thd", "qv", "T", "p", "RH"], "i", f'''
        // equivalent to eqs A11 & A12 in libcloudph++ 1.0 paper
        real_type exponent = {const.Rd} / {const.c_pd};
        real_type pd = pow((rhod[i] * {const.Rd} * thd[i]) / pow({const.p1000}, exponent), 1 / (1 - exponent));
        T[i] = thd[i] * pow((pd / {const.p1000}), exponent);
    
        real_type R = {const.Rv} / (1 / qv[i] + 1) + {const.Rd} / (1 + qv[i]);
        p[i] = rhod[i] * (1 + qv[i]) * R * T[i];
    
        // August-Roche-Magnus formula
        real_type pvs = {const.ARM_C1} * exp(({const.ARM_C2} * (T[i] - {const.T0})) / (T[i] - {const.T0} + {const.ARM_C3}));
    
        RH[i] = (p[i] - pd) / pvs;
    '''.replace("real_type", PrecisionResolver.get_C_type()))

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def temperature_pressure_RH(rhod, thd, qv, T, p, RH):
        PhysicsMethods.__temperature_pressure_RH_body.launch_n(
            T.shape[0], (rhod.data, thd.data, qv.data, T.data, p.data, RH.data))

    __terminal_velocity_body = trtc.For(["values", "radius", "k1", "k2", "k3", "r1", "r2"], "i", '''
        if (radius[i] < r1) {
            values[i] = k1 * radius[i] * radius[i];
        }
        else {
            if (radius[i] < r2) {
                values[i] = k2 * radius[i];
            }
            else {
                values[i] = k3 * pow(radius[i], (real_type)(.5));
            }
        }
        '''.replace("real_type", PrecisionResolver.get_C_type()))

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def terminal_velocity(values, radius, k1, k2, k3, r1, r2):
        k1 = PrecisionResolver.get_floating_point(k1)
        k2 = PrecisionResolver.get_floating_point(k2)
        k3 = PrecisionResolver.get_floating_point(k3)
        r1 = PrecisionResolver.get_floating_point(r1)
        r2 = PrecisionResolver.get_floating_point(r2)
        PhysicsMethods.__terminal_velocity_body.launch_n(values.size(), [values, radius, k1, k2, k3, r1, r2])

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def radius(volume):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dr_dt_MM(r, T, p, RH, kp, rd):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dr_dt_FF(r, T, p, qv, kp, rd, T_i):
        return ""

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def dthd_dt(rhod, thd, T, dqv_dt):
        return ""
