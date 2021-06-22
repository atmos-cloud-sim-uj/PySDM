from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM.physics import si


def test():
    settings = Settings(dt=1 * si.s, n_sd=128, n_substep=1)
    simulation = Simulation(settings)
