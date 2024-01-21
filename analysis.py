import numpy as np
import matplotlib.pyplot as plt

def inspect_spectra(wl,spec_norm,filenames,cutoff=0,alpha=0.3):
    """
    Shows normalised spectra and prints the standard deviation of the residuals.
    The cutoff value is used to print only those spectra with a normalised 
    standard deviation  higher than that value, so that you are able to 
    inspect the worst spectra. Alpha is the plotting transparency.

    Parameters
    ----------
    wl : array-like
        1D array of the input wavelength range, matching the width of spec_norm.

    spec_norm : array-like
        2D array of spectroscopic time-series or residuals. 
        Its width matches the length of wl. 

    filenames : list, tuple
        List of filenames associated with the spectroscopic time-series. Its length
        matches the height of spec_norm.

    cutoff : float, optional
        The limiting standard deviation over which a spectrum is selected.
    
    alpha : float, optional
        The plotting transparency. Set to a small value if selecting many spectra at once.

    Returns
    -------
    sel_filenames : list
        List of selected filenames.


    Examples
    --------
    >>> #Create a mock dataset.
    >>> import numpy as np
    >>> import numpy.random as random
    >>> wl = np.linspace(400,500,10000)
    >>> spec_norm = random.normal(loc=1,scale=0.04,size=(4,len(wl)))
    >>> filenames = [f'file_{int(i)}' for i in np.arange(len(spec_norm))]
    >>> inspect_spectra(wl,spec_norm,filenames,cutoff=0.02,alpha=0.2)
    # filename residual STD
    0 file_0 0.03403698834980731
    1 file_1 0.034387645668130996
    2 file_2 0.03423270128415038
    3 file_3 0.03444028461902215
    ['file_0', 'file_1', 'file_2', 'file_3']
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
    Returns the simple mean in each wlc[i]+-dwl[i] for  each spectrum in spec.

    Parameters
    ----------
    wl : array-like
        1D array of the input wavelength range, matching the width of spec_norm.

    spec : array-like
        2D array of spectroscopic time-series or residuals. 
        Its width matches the length of wl. 

    wlc : list, tuple, array
        List of bin center wavelengths.

    dwl : list, tuple, array
        List of bin half-widths.

    Returns
    -------
    sel_out : array
        Binned spectrum.


    Examples
    --------
    >>> #Create a mock dataset.
    >>> import numpy as np
    >>> import numpy.random as random
    >>> wl = np.linspace(400,500,10000)
    >>> spec = random.normal(loc=1,scale=0.04,size=(2,len(wl)))
    >>> wlc = np.arange(405,408)
    >>> dwl = wlc*0.0+0.1
    >>> bin_wl_range(wl,spec,wlc,dwl)
    [array([0.9981441, 0.9951761]), array([0.99256507, 0.99393875]), array([0.98337412, 0.99355496])]
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
    """
    Selects all spectra that are higher than a threshold value
    in certain ranges. Used to select only those residual spectra in 
    which there is no deviation.


    Parameters
    ----------
    wl : array-like
        1D array of the input wavelength range, matching the width of spec_norm.

    R : array-like
        2D array of residual time-series. 
        Its width matches the length of wl. 
        It is distributed near threshold, which is 1.0 by default.

    wlc : list, tuple, array
        List of bin center wavelengths.

    dwl : list, tuple, array
        List of bin half-widths.

    threshold : float
        Threshold value over which a spectrum is selected.

    plot : bool
        Plot the selected averages in each bin.

    Returns
    -------
    nominal_spec_i : list
        List of indices that satisfy the cutoff in all bins.


    Examples
    --------
    >>> #Create a mock dataset.
    >>> import numpy as np
    >>> import numpy.random as random
    >>> wl = np.linspace(400,500,1000)
    >>> R = random.normal(loc=1,scale=0.04,size=(3,len(wl)))
    >>> wlc = [405,408]
    >>> dwl = [0.01,0.02]
    >>> select_high_spectra(wl,R,wlc,dwl,threshold=0.9)
    [0, 1, 2]
    """
    

    spec_B = bin_wl_range(wl,R,wlc,dwl)

    if plot:
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap('coolwarm')
        for i in range(len(spec_B)):
            plt.plot(spec_B[i],'.',alpha=0.6,color=cmap(i/len(spec_B)))
        plt.show()

    nominal_spec_i = []
    for i in range(len(spec_B[0])):
        deviations = []
        for j in range(len(spec_B)):
            deviations.append(spec_B[j][i])
        if min(deviations) >= threshold:
            nominal_spec_i.append(i)

    return(nominal_spec_i)




