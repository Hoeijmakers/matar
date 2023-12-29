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
import glob
import pdb
import pandas as pd
import tayph.util as ut
from data import test_exists,read_slice
from processing import normslice,inspect_spectra,histfit,cont_fit_line


if __name__ == "__main__":
    # This code will only run if the script is executed directly

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 main.py /input/directory/to/fits/files/")
    inpath = Path(str(sys.argv[1]))

    test_exists(inpath)

    #394.50 396.26 = Al I


    t1=ut.start()
    wl,spec,filename=read_slice(392.21,398.5,inpath)
    spec_norm = normslice(wl,spec,deg=3,reject_regions=[[393.1,393.7],[396.5,397.2]],plot=False)
    # spec_norm=normslice(spec,deg2=7,T=0.3)
    ut.end(t1)

    inspect_spectra(wl,spec_norm,filename,alpha=0.2,cutoff=0.12)
    # histfit(spec_norm[:,10:20])

    # wl_narrow,R,fit=cont_fit_line(wl,spec_norm,range_include=[393.23,393.58],range_ignore=[393.374,393.423],deg=3,plot=True)

    # plt.imshow(R,vmin=0.9,vmax=1.05,aspect='auto')
    # plt.show()

    # for i in range(len(R)):
    #     plt.plot(wl_narrow,R[i],alpha=0.4,color='black',linewidth=0.3)
    # plt.show()

    #This is for sodium.
    # t1=ut.start()
    # wl,spec,filename=read_slice(588.0,592.0,inpath)
    # spec_norm = normslice(wl,spec,deg=3,reject_regions=[],plot=True)
    # # spec_norm=normslice(spec,deg2=7,T=0.3)
    # ut.end(t1)

    # for i in range(len(spec)):
    #     plt.plot(wl,spec_norm[i],color='black',linewidth=0.3,alpha=0.3)
    # plt.show()
