import numpy as np
import matplotlib.pyplot as plt
import astropy.stats as stats
import tayph.functions as fun
from tqdm import tqdm

def histfit(S,plot=False):
    """
    This is the algorithm for identifying the quescent continuum. It works by estimating the mean in each spectral bin
    by fitting the distribution as a Gaussian. However, it only evaluates the histogram partially. It assumes that the
    points are Gaussian distributed from -0.5 sigma to +5 sigma, takes the histogram of those spectra, and then 



    Parameters
    ----------
    S : array-like
        Array to which to fit a Gaussian distribution.

    plot : bool
        Plotting the histogram and the fit, or not.
        For diagnostic purposes as this function is typically wrapped in some loop.


    Returns
    -------
    fitted_mean : float
        The mean of the distrubution.


    Examples
    --------
    >>> #Create a mock dataset with outliers in the negative direction.
    >>> import numpy.random as random
    >>> S = random.normal(loc=1,scale=0.04,size=(50,50))
    >>> histfit(S)
    0.9959915633087365
    >>> S[0:10] = 0.0 #Set 20% of the data to be 0.0 Far far away from the mean.
    >>> histfit(S)
    0.9944485930052719
    """
    std = stats.mad_std(S,ignore_nan=True)
    m = np.nanmedian(S)

    b1=m-0.5*std
    b2=m+5*std
    n_bins = np.min([np.max([10,int(len(S)/20)]),50])#Keep the number of bins between 10 and 50.

    hist,bin_edges = np.histogram(S,bins=n_bins,range=[b1,b2])
    bin_centers = (bin_edges[0:-1]+bin_edges[1:])/2
    fit,errors = fun.gaussfit(bin_centers,hist,nparams=3,startparams=[np.max(hist),m,std])

    if plot:
        plt.plot(bin_centers,hist,'.')
        b_hi = np.linspace(b1,b2,1000)
        plt.plot(b_hi,fun.gaussian(b_hi,*fit))
        plt.axvline(m)
        plt.axvline(fit[1])
        plt.show()
    return(fit[1])

def line_core_fit(wl,spec_norm,range_include=[],range_ignore=[],stepsize=0.01,deg=3,plot=False):
    """
    This fits a gaussian plus polynomial (with degree deg) to a line core in a series of spectra.
    The spectra are first binned down to a stepsize (same units as wl) and then fit using
    tayph.gaussfit. The fit is only to be done on a narrow range (your line core), which is set by 
    the two-element list range_include. A region within that inclusion region can be excluded with the
    two-element range_ignore list. For example if there is a noisy part in the very center, or some
    narrow component you want to avoid (e.g. telluric or ISM or emission).

    The value in each bin is determined by fitting the histogram usting histfit above, 
    ensuring that negative outliers do not participate.



    Parameters
    ----------
    wl : array-like
        1D array of the input wavelength range, matching the width of spec_norm.

    spec_norm : array-like
        2D array of spectral time-series. 
        Its width matches the length of wl. 

    range_include : list, tuple, optional
        A two-element iterable that contains the minimum and maximum wavelength over 
        which to carry out the fit. The whole range of wl is used by default.

    range_ignore : list, tuple, optional
        A two-element iterable that contains a minimum and a maximum wavelength
        to ignore when fitting, used to optionally exclude e.g. the very line core.

    stepsize : float, optional
        The width over which to sample the spectrum (i.e. binsize) when computing the
        histogram, in nm.

    deg : int, optional
        The degree of the polynomial continuum of the gaussian fit.

    plot: bool, optional
        Plotting for diagnostic check.


    Returns
    -------
    wl : array
        The wavelength array over which the fit was done.

    spec_norm : array
        The spectrum array over the range on which the fit was done.

    residual : array
        The spectrum with the gaussian fit divided out.

    fit : array
        The gaussian itself, with length equal to wl.


    Examples
    --------
    >>> #Create a mock dataset with outliers in the negative direction.
    >>> import numpy.random as random
    >>> import tayph.functions as fun
    >>> import numpy as np
    >>> wl = np.arange(400,401,0.002)
    >>> S = random.normal(loc=1,scale=0.04,size=(400,len(wl)))
    >>> S+=fun.gaussian(wl,*[-0.2,400.5,0.1,1.0])
    >>> wl_n,S_n,R,F = line_core_fit(wl,S,range_include=[400.2,400.8],range_ignore=[400.48,400.52],deg=1,plot=False,stepsize=0.02)
    """
    
    wl=wl*1.0
    spec_norm = spec_norm*1.0

    if len(range_include)>0:
        spec_norm=spec_norm[:,(wl>min(range_include))&(wl<max(range_include))]
        wl = wl[(wl>min(range_include))&(wl<max(range_include))] 

    X,Y = [],[]
    N = int((max(wl)-min(wl))/stepsize)
    Ns = int(len(wl)/N)
    if Ns < 1:
        Ns = 1
    print(f'Width: {N*stepsize} nm. N points: {N}. Stride: {Ns} samples.')

    for i in tqdm(range(0,len(wl)-Ns-1,Ns)):
        if np.mean(wl[i:i+Ns]) < np.min(range_ignore) or np.mean(wl[i:i+Ns]) >np.max(range_ignore):
            X.append(np.mean(wl[i:i+Ns]))
            Y.append(histfit(spec_norm[:,i:i+Ns]))



    fit,errors = fun.gaussfit(np.array(X),np.array(Y),nparams=4+deg,startparams=[-1*(np.max(X)-np.min(X)),np.mean(wl),np.std(wl)]+[0]*deg)

    if plot:
        for i in range(len(spec_norm)):
            plt.plot(wl,spec_norm[i],alpha=0.2,linewidth=0.3,color='black')



        plt.plot(X,Y,'o',color='red')
        plt.plot(wl,fun.gaussian(wl,*fit),color='gold')
        plt.show()
    return(wl,spec_norm,spec_norm/fun.gaussian(wl,*fit),fun.gaussian(wl,*fit))





def normslice(wl,spec,error,reject_regions=[],deg=3,plot=True):
    """
    This fits a simple polynomial to the vertical-average-residuals of degree deg.
    reject_regions can be set to a list of tuples that describe the minimum and maximum
    range over which to set the polynomial fitting weights to zero, by means of rejection.
    Otherwise, the weights are set to the square root of the mean spectrum.
 
       Parameters
    ----------
    wl : array-like
        1D array of the input wavelength range, matching the width of spec_norm.

    spec : array-like
        2D array of spectra time-series. 
        Its width matches the length of wl. 

    reject regions : list, tuple, optional
        An iterable of two-element iterables that contains the minimum and maximum wavelengths  
       to be ignored when fitting the continuum. Used to reject line regions.

    deg : int, optional
        The degree of the polynomial fit.

    plot: bool, optional
        Plotting for diagnostic check.


    Returns
    -------
    residuals : array
        The continuum-normalized spectrum.



    Examples
    --------
    >>> #Create a mock dataset with outliers in the negative direction.
    >>> import numpy.random as random
    >>> import tayph.functions as fun
    >>> import numpy as np
    >>> wl = np.arange(400,401,0.002)
    >>> S = random.normal(loc=1,scale=0.04,size=(400,len(wl)))
    >>> S+=fun.gaussian(wl,*[-0.2,400.5,0.1,-200,0.5])
    >>> S_norm = normslice(wl,S,reject_regions=[[400.2,400.25],[400.7,400.75]],deg=1,plot=False)
    """
    M = np.nanmean(spec,axis=0)

    if plot: fig,ax = plt.subplots(2,1,sharex=True)

    w = np.sqrt(M)
    w_nominal = np.sqrt(M)
    p = spec*0.0
    p_nominal = spec*0.0
    for bin in reject_regions:
        w[(wl>bin[0])&(wl<bin[1])] = 0
        if plot:
            ax[0].axvspan(bin[0], bin[1], alpha=0.5, color='red')
            ax[1].axvspan(bin[0], bin[1], alpha=0.5, color='red')

    fit = np.polyfit(wl,(spec/M).T,deg,w=w).T
    if len(reject_regions) != 0:
        fit2 = np.polyfit(wl,(spec/M).T,deg,w=w_nominal).T

    for i in range(len(spec)):
        p[i] = np.poly1d(fit[i])(wl)
        if len(reject_regions) != 0:
            p_nominal[i] = np.poly1d(fit2[i])(wl)


    if plot:
        ax[0].plot(wl,M,label='Mean spectrum')
        ax[1].plot(wl,spec[10]/M,label='Residual')

        if len(reject_regions) != 0:
            ax[1].plot(wl,p[10],label='Polyfit with masked weights')
            ax[1].plot(wl,p_nominal[10],label='Polyfit without masked weights')
        else:
            ax[1].plot(wl,p[10],label='Polyfit, no masked weights set.')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    return(spec/p/np.mean(M),error/p/np.mean(M))
