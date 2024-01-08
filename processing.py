import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.stats as stats
import tayph.functions as fun
from tqdm import tqdm

def histfit(S,plot=False):
    """
    This is the algorithm for identifying the quescent continuum. It works by estimating the mean in each spectral bin
    by fitting the distribution as a Gaussian. However, it only evaluates the histogram partially. It assumes that the
    points are Gaussian distributed from -0.5 sigma to +5 sigma, takes the histogram of those spectra, and then 
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
    """

    
    wl=wl*1.0
    spec_norm = spec_norm*1.0

    if len(range_include)>0:
        spec_norm=spec_norm[:,(wl>min(range_include))&(wl<max(range_include))]
        wl = wl[(wl>min(range_include))&(wl<max(range_include))] 

    X,Y = [],[]
    N = int((max(wl)-min(wl))/stepsize)
    Ns = int(len(wl)/N)
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
    return(wl,spec_norm/fun.gaussian(wl,*fit),fun.gaussian(wl,*fit))






    



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




def normslice(wl,spec,reject_regions=[],deg=3,plot=True):
    """
    This fits a simple polynomial to the vertical-average-residuals of degree deg.
    reject_regions can be set to a list of tuples that describe the minimum and maximum
    range over which to set the polynomial fitting weights to zero, by means of rejection.
    Otherwise, the weights are set to the square root of the mean spectrum.
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
    # print(np.sum(np.abs(p_nominal[10]-p[10])))
        plt.show()

    return(spec/p/np.mean(M))