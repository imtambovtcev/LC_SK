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

initial = Path('/media/ivan/64E21891E2186A16/LC_SK/new_spx/-0.1_5/t/1/')
directory = Path('/media/ivan/64E21891E2186A16/LC_SK/new_spx/-0.1_5/t/10')

print(f'{initial = }')
print(f'{directory = }')

make_map_from_map(save_dir=directory,
                   ref_dir=initial,
                   initial_period_N=1,
                   period_N=10,
                   max_steps_from_minimum=4,
                   z_max_proj=np.infty,
                   reverse=False,
                   max_period=500)

map_info.map_info(directory)