from pathlib import Path
import sys
import numpy as np
from data import test_exists,read_spectrum,read_slice
from processing import normslice
import tayph.operations as ops
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from analysis import read_linelists,plot_labels,select_in_waveband


#This reads and cleans a slice between two closely separated wavelengths and assigns a center wavelength to compute RVs relative to.
if __name__ == "__main__":
    if not len(sys.argv) > 2:
        raise Exception("Call as python3 step_4_save_slice.py /input/directory/to/fits/files/spectra.h5 /input/directory/to/fits/files/clean_spectra.h5 /path/to/outfile.fits minwl maxwl dv")
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))
    outfile = Path(str(sys.argv[3]))
    min_wl = ops.vactoair(float(sys.argv[4]))
    max_wl = ops.vactoair(float(sys.argv[5]))
    c_wl = float(sys.argv[6])
    if len(sys.argv)>7:
        dv = float(sys.argv[7])
    else:
        dv = 0.0
    if len(sys.argv)>8:
        reject_regions=[ops.airtovac(np.array([float(sys.argv[8]),float(sys.argv[9])]))]

    c = 2.99792458e5#km/s
    print(f'shifting by {dv} km/s')
    c_wl *= (1+dv/c)

    test_exists(inpath_1)
    test_exists(inpath_2)

    #First we determine the start and end wavelengths and the slice width
    wl,spec_slice,filename,mjd,exptime=read_slice(min_wl,max_wl,inpath_1)
    wl = ops.airtovac(wl)
    void,void,Q_filename,void,void=read_slice(min_wl,max_wl,inpath_2)
    spec_norm = normslice(wl,spec_slice,deg=2,plot=True,reject_regions=reject_regions)


    #Determine which are the clean spectra
    Q_spec_indices = []
    for i in range(len(filename)):
        if filename[i] in Q_filename:
            Q_spec_indices.append(i)


    mean_clean_spec = np.nanmedian(spec_norm[Q_spec_indices],axis=0)
    R = spec_norm/mean_clean_spec

    RV = c*(wl-c_wl)/wl

    plt.imshow(R)
    plt.show()


    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(data=spec_norm,name='spec_norm'))
    new_hdul.append(fits.ImageHDU(data=wl,name='wl (vac)'))
    new_hdul.append(fits.ImageHDU(data=R,name='residuals'))
    new_hdul.append(fits.ImageHDU(data=mean_clean_spec,name='mean_clean_spec'))
    new_hdul.append(fits.ImageHDU(data=RV,name='RV'))
    new_hdul.append(fits.ImageHDU(data=mjd,name='MJD'))
    new_hdul.append(fits.ImageHDU(data=exptime,name='EXPTIME'))
    new_hdul.writeto(outfile,overwrite=True)

    plt.plot(RV,mean_clean_spec)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel("Relative flux")
    plt.axvline(0.0,color='red')
    plt.title(f'Selected slice saved to {str(outfile)}')
    plt.show()








