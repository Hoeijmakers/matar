from pathlib import Path
import sys
import numpy as np
from data import test_exists,read_spectrum,read_slice
from processing import normslice
import tayph.operations as ops
import pdb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from analysis import read_linelists,plot_labels,select_in_waveband

if __name__ == "__main__":
    if not len(sys.argv) > 2:
        raise Exception("Call as python3 step_4_line_identification.py /input/directory/to/fits/files/spectra.h5 /input/directory/to/fits/files/clean_spectra.h5 linelist_paths.dat minwl maxwl optional_dv")
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))
    inpath_3 = Path(str(sys.argv[3]))
    min_wl = float(sys.argv[4])
    max_wl = float(sys.argv[5])
    test_exists(inpath_1)
    test_exists(inpath_2)
    test_exists(inpath_3)



    if len(sys.argv) > 6:
        dv = float(sys.argv[6])
    else:
        dv = 0.0

    labels,wlm,sm = read_linelists(inpath_3)
    print(f'The following spectra are in {str(inpath_3)}:')
    print(labels)


    #First we determine the start and end wavelengths and the slice width
    wl,spec_slice,filename,void,void=read_slice(min_wl,max_wl,inpath_1)
    wl = ops.airtovac(wl)
    void,void,Q_filename,void,void=read_slice(min_wl,max_wl,inpath_2)
    spec_norm = normslice(wl,spec_slice,deg=3,plot=False)

    #Determine which are the clean spectra
    Q_spec_indices = []
    for i in range(len(filename)):
        if filename[i] in Q_filename:
            Q_spec_indices.append(i)


    mean_clean_spec = np.nanmedian(spec_norm[Q_spec_indices],axis=0)
    R = spec_norm/mean_clean_spec



    plt.plot(wl,mean_clean_spec)
    # plot_labels(['species1','species2','etc'],labels,wlm,sm,dv=dv)
    plt.xlim(min(wl),max(wl))
    plt.ylim(0.0,np.max(mean_clean_spec))
    plt.show()








