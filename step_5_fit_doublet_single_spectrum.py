from data import test_exists
from analysis import load_order_RV_range, load_spectrum_for_fitting, find_spectra_in_mjd
from fitting import fit_lines,supersample, gaussian_skewed, get_bestfit_params,setup_numpyro, read_prior, save_fit_output
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
from matplotlib.backends.backend_pdf import PdfPages
import pdb
from tqdm import tqdm
import astropy.constants as const
from numpy.random import uniform
import os
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_5_plot_slice.py path/to/spectra.h5 path/to/configfile.dat")
    
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))
    inpath_3 = Path(str(sys.argv[3]))

    test_exists(inpath_1)
    test_exists(inpath_2)

    C = np.loadtxt(inpath_2)
    wlc1 = C[0]
    wlc2 = C[1]
    drvmin  = C[2]
    drvmax = C[3]
    vsys = C[4]
    mjd_target = C[5] #The MJD date within +1 day we are going to be fitting spectra.
    N = int(C[6]) #Which spectrum we're going to fit in this night.
    oversampling = int(C[7])
    cpu_cores = int(C[8])
    setup_numpyro(cpu_cores)
    nwarmup  = int(C[9])
    nsamples = int(C[10])
    plot_trigger = int(C[11])
    c = const.c.to('km/s').value


    if len(sys.argv)>4:
        outpath = Path(str(sys.argv[4]))
    else:
        outpath = Path('fit_results/')
    outpath = outpath/str(mjd_target)

    if outpath.exists() == False:
        os.makedirs(outpath)

    filelist = find_spectra_in_mjd(inpath_1,mjd_target,ignore_empty=False)
    v_out,xs_out,y_out,yerr_out,orders_ret,mjd_ret = load_spectrum_for_fitting(inpath_1,filelist[N],[wlc1,wlc2],drvmin=drvmin,drvmax=drvmax,vsys=vsys,oversampling=5)

    bounds = read_prior(inpath_3)

    x_in,y_in,yerr_in = [],[],[]#In, for what goes into the fitting routine.
    for i in range(len(xs_out)):
        for j in range(len(xs_out[i])):
            x_in.append(     xs_out[i][j])
            y_in.append(      y_out[i][j])           
            yerr_in.append(yerr_out[i][j])

    samples,model = fit_lines(x_in,y_in,yerr_in,bounds,cpu_cores=cpu_cores,oversample=5,absorption=True,progress_bar=True,nwarmup=nwarmup,nsamples=nsamples,pass_supers=True,plotname='test',plot=plot_trigger)



    #Now we proceed to shape the data to save it.

    dataslices=[]
    for i in range(len(x_in)):
        D = np.zeros((3,len(y_in[i])))
        D[0]=x_in[i].mean(axis=1)
        D[1]=y_in[i]
        D[2]=yerr_in[i]
        dataslices.append(D)

    save_fit_output(outpath/(Path(filelist[N]).stem+'.nc'),samples,dataslices,attrs={'mjd':mjd_ret})


    # p,e = get_bestfit_params(samples)
    # for k in samples.keys():
    #     print(f'{k}: {p[k]} +- {e[k]}')
