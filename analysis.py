import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.stats as stats
import tayph.functions as fun
from tqdm import tqdm

def inspect_spectra(wl,spec_norm,filenames,cutoff=0,alpha=0.3):

    """
    This shows normalised spectra and prints the standard deviation of the residuals.
    The cutoff value is used to print only those spectra with a normalised 
    standard deviation  higher than that value, so that you are able to 
    inspect the worst spectra. Alpha is the plotting transparency.
    """
    R = spec_norm/np.nanmean(spec_norm,axis=0)
    S = np.nanstd(R,axis=1)
    print('#','filename','residual STD')
    for i in range(len(spec_norm)):
        if S[i]>cutoff: 
            print(i,filenames[i],S[i])
            plt.plot(wl,spec_norm[i],color='black',linewidth=0.3,alpha=alpha)
    plt.show()
    return


def bin_wl_range(wl,spec,wlc,dwl):
    """
    This bins a 2D spectrum (spec) between wlc+-dwl.
    wlc can be passed both as a list and as a float.
    """
    if type(wlc) == float:
        wlc=[wlc]
    if type(dwl) == float:
        dwl=[dwl]
    

    spec_out = []
    for i in range(len(wlc)):
        wlmin = wlc[i]-dwl[i]
        wlmax = wlc[i]+dwl[i]
        spec_sel = spec[:,(wl>wlmin)&(wl<wlmax)]
        spec_bin = np.nanmean(spec_sel,axis=1)
        if len(wlc) == 1:
            return(spec_bin)
        else:
            spec_out.append(spec_bin)
    return(spec_out)




