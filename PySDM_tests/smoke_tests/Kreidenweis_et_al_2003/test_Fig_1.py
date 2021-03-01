from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from matplotlib import pyplot
import numpy as np
import pytest


@pytest.fixture(scope='session')
def example_output():
    settings = Settings(n_sd=1, dt=1*si.s)
    simulation = Simulation(settings)
    output = simulation.run()
    return output

Z_CB = 196 * si.m


class TestFig1:
    @staticmethod
    def test_a(example_output, plot=True):
        # Plot
        if plot:
            name = 'ql'
            #prod = simulation.core.products['ql']
            pyplot.plot(example_output[name], np.asarray(example_output['t']) - Z_CB * si.s)
            #pyplot.xlabel(f"{prod.name} [{prod.unit}]")  # TODO
            pyplot.ylabel(f"time above cloud base [s]")
            pyplot.grid()
            pyplot.show()

        # Assert
        assert (np.diff(example_output['ql']) >= 0).all()

    @staticmethod
    def test_b(example_output, plot=True):
        # Plot
        if plot:
            pyplot.plot(
                np.asarray(example_output['aq_S_IV_ppb']) + np.asarray(example_output['gas_S_IV_ppb']),
                np.asarray(example_output['t']) - Z_CB * si.s)
            pyplot.xlim(0, .21)
            pyplot.show()

        # Assert
        # assert False  TODO

    @staticmethod
    def test_c(example_output, plot=True):
        if plot:
            pyplot.plot(example_output['pH'], np.asarray(example_output['t']) - Z_CB * si.s)
            pyplot.show()

        #  assert False  TODO