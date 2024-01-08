import os
import numpy as np
from pathlib import Path
import astropy.io.fits as fits
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import tayph.functions as fun
import tayph.operations as ops
import sys
import tayph.util as ut
from data import test_exists,read_slice
from processing import normslice,inspect_spectra,histfit,cont_fit_line


if __name__ == "__main__":
    # This code will only run if the script is executed directly

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 main.py /input/directory/to/fits/files/")
    inpath = Path(str(sys.argv[1]))

    test_exists(inpath)

