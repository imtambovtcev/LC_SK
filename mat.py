import numpy as np
import sys
import os
import shutil
import glob

import magnes
import magnes.graphics

import matplotlib.pyplot as plt
import map_file_manager as mfm
import map_info
import scipy.interpolate
from pathlib import Path

from termcolor import colored
from minimize import *

initial = Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/initials/alt/20/1.npz')
directory = Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/new_spx/alt_20/1')
print(f'{initial = }')
print(f'{directory = }')

make_map_from_file(save_dir=directory,
                   KDbulk_list=np.linspace(0, -0.3, 31), #(0, -2.,41)(0, -0.5,11)
                   KDsurf_list=np.linspace(0, 20,11), #(0,100,51)(0, 20,21)
                   ref=initial,
                   period_N=1,
                   max_steps_from_minimum=5,
                   z_max_proj=np.infty,
                   reverse=True,
                   max_period=500)

map_info.map_info(directory)