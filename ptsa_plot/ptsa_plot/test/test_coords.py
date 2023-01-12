import unittest
from numpy import pi
import numpy as np
import numpy.random as npr
from numpy.testing import assert_almost_equal
from ptsa_plot import coords


class TestCoords(unittest.TestCase):
    NUM_TRIALS = 20

    def test_deg2rad(self):
        # single value
        deg = npr.uniform(-359.9, 359.9)
        assert_almost_equal(coords.deg2rad(deg), pi*deg/180.)

        # Many values
        degs = npr.uniform(-359.9, 359.9, (100,))
        for deg in degs:
            assert_almost_equal(coords.deg2rad(deg), pi * deg / 180.)

    def test_rad2deg(self):
        # Single value
        rad = npr.uniform(-2*pi, 2*pi)
        assert_almost_equal(coords.rad2deg(rad), 180*rad/pi)

        # Many values
        rads = npr.uniform(-2*pi, 2*pi, (100,))
        for rad in rads:
            assert_almost_equal(coords.rad2deg(rad), 180*rad/pi)

    def test_pol2cart(self):
        angle, r = npr.uniform(-2*pi, 2*pi), npr.uniform()
        x, y = r * np.cos(angle), r * np.sin(angle)
        z = npr.uniform(-1, 1)
        assert_almost_equal(coords.pol2cart(angle, r), (x, y))
        assert_almost_equal(coords.pol2cart(angle, r, z=z), (x, y, z))

        angle, r = npr.uniform(-360, 360), npr.uniform()
        x, y = r * np.cos(pi * angle / 180.), r * np.sin(pi * angle / 180.)
        z = npr.uniform(-1, 1)
        assert_almost_equal(coords.pol2cart(angle, r, radians=False), (x, y))
        assert_almost_equal(coords.pol2cart(angle, r, z, radians=False), (x, y, z))

    def test_cart2pol(self):
        x, y, Z = npr.uniform(-1, 1, (3,))
        z = x + y * 1.j

        assert_almost_equal(
            coords.cart2pol(x, y), (np.angle(z), np.hypot(x, y))
        )
        assert_almost_equal(
            coords.cart2pol(x, y, Z), (np.angle(z), np.hypot(x, y), Z)
        )

        assert_almost_equal(
            coords.cart2pol(x, y, radians=False), (np.angle(z, deg=True), np.hypot(x, y))
        )
        assert_almost_equal(
            coords.cart2pol(x, y, Z, radians=False), (np.angle(z, deg=True), np.hypot(x, y), Z)
        )
