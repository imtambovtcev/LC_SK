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

initial = Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/initials/spx/20/1.npz')
directory = Path('/media/ivan/64E21891E2186A16/Users/vano2/Documents/LC_SK/new_spx/20/1')

make_map_from_file(save_dir=directory,
                   KDbulk_list=np.linspace(0, -2,41),
                   KDsurf_list=np.linspace(0, 30,31), #(0,100,51)
                   ref=initial,
                   period_N=1,
                   max_steps_from_minimum=5,
                   z_max_proj=0.25,
                   reverse=True,
                   max_period=500)

map_info.map_info(directory)