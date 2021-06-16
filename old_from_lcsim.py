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

ref_file=Path('/home/ivan/LC_SK/initials/cone/cone1/20.npz')
initial = Path('/home/ivan/LC_SK/skt/skyrmion')
directory = Path('/home/ivan/LC_SK/skt/cone')

print(f'{initial = }')
print(f'{directory = }')

make_map_from_map(directory=initial,save_directory=directory,ref_file=ref_file,state_name='cone',maxiter=10000)

map_info.map_info(directory)