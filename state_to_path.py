import numpy as np
from termcolor import colored
import sys
import os

import time
import matplotlib
from pathlib import Path
from shutil import copyfile
import map_file_manager as mfm
from scipy.interpolate import CubicSpline
import utilities
import magnes

def state_to_path(directory):
    directory=Path(directory)
    filelist=[x for x in directory.iterdir() if x.suffix == '.npz']
    for file in filelist:
        utilities.state_to_path(file)

if __name__ == "__main__":
    directory = Path('./') if len(sys.argv) <= 1 else Path(sys.argv[1])
    state_to_path(directory)