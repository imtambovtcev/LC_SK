#!/usr/bin/env python
import numpy as np
import sys
import os

import os.path
import magnes
import magnes.graphics
import magnes.utils
import matplotlib.pyplot as plt
from pathlib import Path
import map_file_manager
from scipy.interpolate import CubicSpline
import scipy.interpolate
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator


def initial(vector_list=[[0, 0, 0, 0, 0, 1], [100, 0, 0, 0, 0, 1], [0, 100, 0, 0, 0, 1], [100, 100, 0, 0, 0, 1],
                         [0, 0, 2, 0, 0, 1], [100, 0, 2, 0, 0, 1], [0, 100, 2, 0, 0, 1], [100, 100, 2, 0, 0, 1],
                         [25, 0, 1, 0, 0, 1], [0, 25, 1, 0, 0, 1], [75, 0, 1, 0, 0, 1], [0, 75, 1, 0, 0, 1],
                         [25, 100, 1, 0, 0, 1], [100, 25, 1, 0, 0, 1], [75, 100, 1, 0, 0, 1], [100, 75, 1, 0, 0, 1],
                         [0, 50, 1, 0, 0, 1], [50, 0, 1, 0, 0, 1], [100, 50, 1, 0, 0, 1], [50, 100, 1, 0, 0, 1],
                         [16, 15, 1, 0, 0, 1], [15, 85, 1, 0, 0, 1], [85, 15, 1, 0, 0, 1], [85, 85, 1, 0, 0, 1],
                         [25, 50, 1, 0, -1, 0], [50, 25, 1, 1, 0, 0], [50, 75, 1, -1, 0, 0], [75, 50, 1, 0, 1, 0],
                         [35, 35, 1, 1, -1, -1], [65, 25, 1, 1, 1, 1], [35, 65, 1, -1, -1, 0], [65, 65, 1, -1, 1, 0],
                         [50, 50, 1, 0, 0, -1]], size=[100, 100, 3]):
    vector_list = np.array(vector_list, dtype=np.float64)
    vector_norm = np.linalg.norm((np.array(vector_list[:, 3:])), axis=1).reshape(-1, 1)
    vector_list[:, 3:] = vector_list[:, 3:] / vector_norm
    vector_list = np.array(vector_list)

    x = np.linspace(0, size[0], size[0])
    y = np.linspace(0, size[1], size[1])
    z = np.linspace(0, size[2], size[2])

    s = np.zeros(size + [1, 3])
    interpolator=LinearNDInterpolator
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    interp = interpolator(list(vector_list[:, :3]), vector_list[:, 3])
    s[:, :, :, 0, 0] = interp(X, Y, Z)
    interp = interpolator(list(vector_list[:, :3]), vector_list[:, 4])
    s[:, :, :, 0, 1] = interp(X, Y, Z)
    interp = interpolator(list(vector_list[:, :3]), vector_list[:, 5])
    s[:, :, :, 0, 2] = interp(X, Y, Z)

    return s
