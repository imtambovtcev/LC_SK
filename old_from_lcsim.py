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

if __name__ == "__main__":
    load_file = Path(sys.argv[1])
    save_file = Path(sys.argv[2])
    print(f'{save_file = }')
    load=magnes.io.load(str(load_file))
    Path=load['PATH']
    print(f'{Path.shape = }')
    system = magnes.SquareLattice(Path.shape, bc=[magnes.BC.FREE, magnes.BC.FREE, magnes.BC.FREE], J=0, DM_rho=0, DM_phi=0)

