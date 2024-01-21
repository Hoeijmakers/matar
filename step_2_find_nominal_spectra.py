import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import tayph.util as ut
from data import test_exists,read_slice
from processing import normslice,line_core_fit
from analysis import select_high_spectra


if __name__ == "__main__":

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 main.py /input/directory/to/fits/files/ optional_filename.h5 optional_out_filename.txt")
    datafolder = Path(str(sys.argv[1]))
    test_exists(datafolder)
    if len(sys.argv) > 2:
        inpath = datafolder/str(sys.argv[2])
    else:
        inpath = datafolder/'spectra.h5'
    test_exists(inpath)

    if len(sys.argv) > 3:
        outpath = Path(str(sys.argv[3]))
    else:
        outpath = datafolder/'selected_spectra.txt'


    #This reads the data saved in the dataframe. For small slices it does so very quickly.
    t1=ut.start()
    wl,spec,filename,mjd=read_slice(392.21,398.5,inpath)
    spec_norm = normslice(wl,spec,deg=3,reject_regions=[[393.1,393.7],[396.5,397.2]],plot=False)
    ut.end(t1)


    #We then fit the line core, excluding the narrow range around the line center.
    wl_narrow,spec_narrow,R,fit=line_core_fit(wl,spec_norm,range_include=[393.23,393.58],range_ignore=[393.374,393.423],deg=5,plot=False)
    #This is done after the data have been made into a dataframe, and after removing from the input folder some spectra 
    #that may be bad, for which the inspect_spectra function has been written.


    #We plot the residuals, and select ranges within which to test the flux.
    plt.pcolormesh(wl_narrow,np.arange(len(R)),R,vmin=0.9,vmax=1.05)
    wlc = [393.295,393.344,393.358,393.369,393.379,393.410,393.425,393.445,393.4618,393.4847,393.52]
    dwl = [0.01,0.005,0.005,0.005,0.003,0.005,0.005,0.005,0.005,0.01,0.015]
    for i in range(len(wlc)):
        plt.axvspan(wlc[i]-dwl[i],wlc[i]+dwl[i],color='red',alpha=0.2)
    plt.show()


    #We then select all spectra from the series where the flux is high in these bands.
    nominal_spec_i = select_high_spectra(wl_narrow,R,wlc,dwl,threshold=0.925,plot=False)

    #We determine the mean spectrum of those and plot it.
    mean_clean_spec = np.nanmedian(spec_norm[nominal_spec_i],axis=0)
    for i in nominal_spec_i:
        plt.plot(wl,spec_norm[i],color='black',alpha=0.05)
    plt.plot(wl,mean_clean_spec,color='red',label='Median spectrum')
    plt.legend()
    plt.show()


    #We compute the residuals. This is our 'clean' data.
    R2 = spec_norm/mean_clean_spec
    plt.pcolormesh(wl,np.arange(len(R2)),R2,vmin=0.9,vmax=1.05,cmap='copper')
    plt.title('Residuals after removing median of selected clean spectra')
    plt.show()


    with open(outpath,'w') as fp:
        for item in filename[nominal_spec_i]:
            fp.write(f"{item}\n")
    print(f'Assigned {int(len(nominal_spec_i))} files in {str(outpath)}')

