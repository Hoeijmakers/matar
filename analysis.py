import numpy as np
import matplotlib.pyplot as plt

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
    sel_filenames=[]
    for i in range(len(spec_norm)):
        if S[i]>cutoff: 
            print(i,filenames[i],S[i])
            plt.plot(wl,spec_norm[i],color='black',linewidth=0.3,alpha=alpha)
            sel_filenames.append(filenames[i])
    plt.show()
    return(sel_filenames)


def bin_wl_range(wl,spec,wlc,dwl):
    """
    This bins a 2D spectrum (spec) between wlc+-dwl.
    wlc can be passed both as a list and as a float.
    Can also be used to bin residuals to check for systematic
    deviations.
    Returns the mean value in each wlc[i]+-dwl[i] for each spectrum in spec.
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



def select_high_spectra(wl,R,wlc,dwl,threshold=0.95,plot=False):
    """This selects all spectra that are higher than a threshold value
    in certain ranges. Used to select only those residual spectra in 
    which there is no deviation."""
    import matplotlib.cm
    spec_B = bin_wl_range(wl,R,wlc,dwl)


    if plot:
        cmap = matplotlib.cm.get_cmap('coolwarm')
        for i in range(len(spec_B)):
            plt.plot(spec_B[i],'.',alpha=0.6,color=cmap(i/len(spec_B)))
        plt.show()


    # for i in range(len(spec_B)):
    #     plt.hist(spec_B[i],bins=np.arange(0.0,1.2,0.02),alpha=0.5)
    # plt.show()

    nominal_spec_i = []
    for i in range(len(spec_B[0])):
        deviations = []
        for j in range(len(spec_B)):
            deviations.append(spec_B[j][i])
        if min(deviations) >= threshold:
            nominal_spec_i.append(i)

    return(nominal_spec_i)




