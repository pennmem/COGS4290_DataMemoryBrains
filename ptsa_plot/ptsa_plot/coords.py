"""Coordinate system conversion utilities."""

import numpy as np


def deg2rad(degrees):
    """Convert degrees to radians."""
    return degrees / 180. * np.pi


def rad2deg(radians):
    """Convert radians to degrees."""
    return radians / np.pi * 180.


def pol2cart(theta, radius, z=None, radians=True):
    """Converts corresponding angles (theta), radii, and (optional) height (z)
    from polar (or, when height is given, cylindrical) coordinates
    to Cartesian coordinates x, y, and z.
    Theta is assumed to be in radians, but will be converted
    from degrees if radians==False.

    """
    if radians:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    else:
        x = radius * np.cos(deg2rad(theta))
        y = radius * np.sin(deg2rad(theta))
    if z is not None:
        return x, y, z
    else:
        return x, y


def cart2pol(x, y, z=None, radians=True):
    """Converts corresponding Cartesian coordinates x, y, and (optional) z
    to polar (or, when z is given, cylindrical) coordinates
    angle (theta), radius, and z.
    By default theta is returned in radians, but will be converted
    to degrees if radians==False.

    """
    if radians:
        theta = np.arctan2(y, x)
    else:
        theta = rad2deg(np.arctan2(y, x))
    radius = np.hypot(x, y)
    if z is not None:
        return theta, radius, z
    else:
        return theta, radius
