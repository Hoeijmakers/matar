from pathlib import Path
import sys
import numpy as np
from data import test_exists,read_spectrum,read_slice
from processing import normslice
import math
import pdb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
if __name__ == "__main__":
    if not len(sys.argv) > 2:
        raise Exception("Call as python3 step_3_survey_wl.py /input/directory/to/fits/files/spectra.h5 /input/directory/to/fits/files/selected_spectra.h5")
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))
    test_exists(inpath_1)
    test_exists(inpath_2)

    #First we determine the start and end wavelengths and the slice width
    void,void,filename,void,void=read_slice(392.21,398.5,inpath_1)
    void,void,Q_filename,void,void=read_slice(392.21,398.5,inpath_2)
    wl,fx,err,hdr = read_spectrum(filename[0].decode('unicode_escape'))
    wl_start,wl_end,dwl = min(wl),660,5

    #Determine which are the clean spectra
    Q_spec_indices = []
    for i in range(len(filename)):
        if filename[i] in Q_filename:
            Q_spec_indices.append(i)



    slices = []
    for wl_0 in tqdm(range(math.ceil(wl_start),wl_end,dwl)):
        slices.append(wl_0)
        wl_slice,spec_slice,filename,mjd,exptime=read_slice(wl_0,wl_0+dwl,inpath_1)
        try:
            spec_norm = normslice(wl_slice,spec_slice,deg=3,plot=False)
        except:
            spec_norm = np.zeros(np.shape(spec_slice))+1.0
        mean_clean_spec = np.nanmedian(spec_norm[Q_spec_indices],axis=0)

        R = spec_norm/mean_clean_spec

        fig=plt.figure(figsize=(20,16))
        gs = GridSpec(8, 1, figure=fig)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1:-2,0])
        ax3 = fig.add_subplot(gs[-2:-1,0])
        ax2 = fig.add_subplot(gs[-1,0])
        ax0.plot(wl_slice,mean_clean_spec)
        ax0.set_xlim(min(wl_slice),max(wl_slice))
        ax2.set_xlim(min(wl_slice),max(wl_slice))
        ax1.pcolormesh(wl_slice,np.arange(len(R)),R,vmin=0.975,vmax=1.0125,cmap='copper')
        ax2.plot(wl_slice,np.mean(R,axis=0))
        ax3.pcolormesh(wl_slice,np.arange(len(Q_spec_indices)),R[Q_spec_indices],vmin=0.975,vmax=1.0125,cmap='copper')
        ax2.set_ylabel('Mean residual)')
        ax2.set_xlabel('Wavelength (nm)')
        ax0.set_ylabel('Avg spec flux')
        ax1.set_ylabel('Spec ID')
        ax0.set_title('Mean spectrum')
        ax1.set_title('Residual')
        ax2.set_title('Mean residual')
        ax3.set_title('Residual of clean spectra')
        ax2.set_ylim(0.994,1.005)
        plt.savefig(f'out_plots/slice_{int(wl_0)}_{wl_0+dwl}.png',dpi=300)
        plt.close()
        # plt.show()
        # pdb.set_trace()

        # R = spec_slice/np.nanmedian(M_spec_slice,axis=0)
        # print(R.nbytes/1e6)








