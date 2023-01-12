import os.path as osp
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')

from ptsa_plot.topo import topoplot


class TestTopo(unittest.TestCase):
    def test_topoplot(self):
        """Just a basic test to ensure that nothing crashes."""
        with open(osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'HCGSN128.dat'), 'r') as sensorfile:
            angles, radii = sensorfile.readlines()
            sensors = (np.r_[np.array([np.float(a)
                                       for a in angles.strip().split()]), 0],
                       np.r_[[np.float(r) for r in radii.strip().split()], 0])

        topoplot(sensors=sensors, values=np.random.randn((129)), labels=range(129), plot_mask='circular')
