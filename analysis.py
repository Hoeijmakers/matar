import numpy as np
import matplotlib.pyplot as plt
from data import read_order
from data import test_exists
from fitting import supersample

from matplotlib.backends.backend_pdf import PdfPages
import pdb
from tqdm import tqdm
import astropy.constants as const
from numpy.random import uniform

def find_spectra_in_mjd(inpath_1,mjd_target,ignore_empty=False):
    """
    This finds all spectra taken within a 24h time interval counted from an mjd date in the saved h5 file,
    and returns their filenames sorted in mjd. It returns an error in case no files are found for the
    requested mjd.
    """
    wl,fx,fxt,err,filelist,mjd,exptime,berv = read_order(0,inpath_1,px_range=[0,2])
    sel = (mjd>mjd_target)&(mjd<mjd_target+1)
    filelist_to_return = filelist[sel]

    if not ignore_empty and len(filelist_to_return) == 0:
        raise Exception(f"No files found for mjd = {mjd_target}")
    for i in range(len(filelist_to_return)):
        filelist_to_return[i] = filelist_to_return[i].decode('utf-8')
    return(filelist_to_return[np.argsort(mjd[sel])])

def load_spectrum_for_fitting(inpath_1,filename,wlcs,drvmin=-100,drvmax=100,vsys=0.0,oversampling=5):
    """
    This loads spectrum segments for fitting. wlcs contains a list of center wavelengths plusminus
    a velocity range drvmin-drvmax. The script then proceeds to pull those ranges out of the spectrum
    file, taking into account the possibiity that this data may exist at an order edge.


    The output are lists (with the same length as wlcs) of the velocity, super-sampled wl, spectrum and error axes, and which orders were returned.
    """


    test_exists(inpath_1)


    c = const.c.to('km/s').value

    drv = np.max(np.abs([drvmin,drvmax]))

    v_out,xs_out,y_out,yerr_out,orders_ret = [],[],[],[],[]
    for wlc in wlcs:
        wl1,RV1,orders_norm1,orders_norm_t1,err1,filelist1,mjd1,t_exp1,berv1,orders_returned1,px_lims_returned1 = load_order_RV_range(inpath_1,wlc*(1+vsys/c),drv+33)#33 km/s of padding is added because we will still be BERV correcting and 33km/s is the maximum possible BERV.
    # wl2,RV2,orders_norm2,orders_norm_t2,err2,filelist2,mjd2,t_exp2,berv2,orders_returned2,px_lims_returned2 = load_order_RV_range(inpath_1,wlc2*(1+vsys/c),drv+33)

        if wlc == wlcs[0]: #Only determine this the first time.
            sel,j = [None,0]

            for f in filelist1:
                if f.decode('utf-8') == filename:
                    sel = int(j*1.0)
                j+=1
    
        if sel is None:
            raise Exception(f'{filename} not found in {inpath_1}')

        mjd_ret = mjd1[sel]
        #This is written to extract down from the data the spectrum that we need to fit, dealing with order overlap:
        wl1_out,RV1_out,orders_norm1_out,orders_norm_t1_out,err1_out = [],[],[],[],[]
        # wl2_out,RV2_out,orders_norm2_out,orders_norm_t2_out,err2_out = [],[],[],[],[]
        for i in range(len(orders_returned1)):
            wl1_out.append(wl1[i][sel])
            RV1_out.append(RV1[i][sel])
            orders_norm1_out.append(orders_norm1[i][sel])
            orders_norm_t1_out.append(orders_norm_t1[i][sel])
            err1_out.append(err1[i][sel])

        #Now we down-select to the spectrum that we want on the RV range that we want.
        xs1,y1,yerr1,v1 = [],[],[],[]
        for i in range(len(orders_returned1)):
            wl_sel = (RV1_out[i]>drvmin)&(RV1_out[i]<drvmax)
            v1.append(RV1_out[i][wl_sel])
            xs1.append(supersample(wl1_out[i][wl_sel],f=oversampling))
            y1.append(orders_norm1_out[i][wl_sel])
            yerr1.append(err1_out[i][wl_sel])

        v_out.append(v1)
        xs_out.append(xs1)
        y_out.append(y1)
        yerr_out.append(yerr1)
        orders_ret.append(orders_returned1)
    return(v_out,xs_out,y_out,yerr_out,orders_ret,mjd_ret)
    


    # for i in range(len(orders_returned2)):
    #     wl2_out.append(wl2[i][sel])
    #     RV2_out.append(RV2[i][sel])
    #     orders_norm2_out.append(orders_norm2[i][sel])
    #     orders_norm_t2_out.append(orders_norm_t2[i][sel])
    #     err2_out.append(err2[i][sel])
    # xs2,y2,yerr2,v2 = [],[],[],[]
    # for i in range(len(orders_returned2)):
    #     wl_sel = (RV2_out[i][N]>drvmin)&(RV2_out[i][N]<drvmax)
    #     v2.append(RV2_out[i][N][wl_sel])
    #     xs2.append(supersample(wl2_out[i][N][wl_sel],f=oversampling))
    #     y2.append(orders_norm2_out[i][N][wl_sel])
    #     yerr2.append(err2_out[i][N][wl_sel])
    # v2_c = np.concatenate(v2)
    # xs2_c = np.concatenate(xs2,axis=0)
    # y2_c = np.concatenate(y2)
    # yerr2_c = np.concatenate(yerr2)





def load_order_RV_range(inpath,wlc,RV,norm=True,bervcor=True):
    """
    This loads an order, does BERV correction, selects within a certain center wavelength and RV range 
    (automatically pulling the correct order(s) out), and doing normalization.
    The wavelength slice within the order(s) that contain the center wavelength has a width of approximately
    2x RV; but the width and center wavelength vary slightly because the wavelength axes of the pixel grid are not
    assumed to be exactly constant.
    
    The returned RV grid has wlc as its center point, but after application of berv correction."""

    from data import read_order, read_exposure
    import sys
    import pdb
    import astropy.constants as const
    #selected_wl,selected_data,selected_err,filelist,mjd,exptime,berv = read_order(n,hdf_file_path,px_range=[],exp=None)
    #selected_wl,selected_data,selected_err,filelist,mjd,exptime,berv = read_exposure(n,hdf_file_path)
    

    #First load in a single exposure to get the approximate wavelength axis of all the orders.
    #We use this to determine what order(s) to select.
    c = const.c.to('km/s').value
    A = read_exposure(0,inpath)
    wl2d = A[0]
    orders_to_return = []
    
    
    r = wl2d - wlc
    r_left = wl2d - wlc*(1-RV/c)
    r_right = wl2d - wlc*(1+RV/c)
    for i in range(len(wl2d)):
        if (min(r[i])<0 and max(r[i])>0) or (min(r_left[i])<0 and max(r_left[i])>0) or (min(r_right[i])<0 and max(r_right[i])>0): #This tests if the wlc or any of the selection boundaries occur in any of the orders.
            orders_to_return.append(i)


    if len(orders_to_return) == 0:
        raise Exception('Error, wlc±RV is not contained in the data!')
    

    #Now, within these orders, we determine what pixel ranges to return.
    px_lims_to_return = []
    
    for i in orders_to_return:
        v = (wl2d[i]-wlc)/wlc * c
        left_px_lim,right_px_lim = np.argmin(np.abs(v+RV)),np.argmin(np.abs(v-RV))
        px_lims_to_return.append([left_px_lim,right_px_lim])


    #Now we read in the actual orders, do bervcor and continuum normalisation, and return as lists.
    out_wl,out_RV,out_fx,out_fxt,out_err,meanflux = [],[],[],[],[],[]
    for i,n in enumerate(orders_to_return): #Can actually only be two, but OK.
        wl,fx,fxt,err,filelist,mjd,exptime,berv = read_order(n,inpath,px_range=px_lims_to_return[i])

        # if norm:
        #     meanflux = np.mean(fx,axis=1)
        # else:
        #     meanflux = 1.0
        if bervcor:
            gamma = 1+(berv/c)
        else:
            gamma = 1
        wl_corr = (wl.T*gamma).T
        v_axis = (wl_corr-wlc) / wlc *c

        out_wl.append(wl_corr)
        out_RV.append(v_axis)
        out_fx.append(fx)#This is the spectra with tellurics corrected.
        out_fxt.append(fxt)#Thats the spectra with tellurics remaining.
        out_err.append(err)

        #Note how we are dividing by the un-bervcorrected meanflux. Preserving the flux of the order in the original range.
    meanflux = np.mean(np.concatenate(out_fx,axis=1),axis=1)
    if norm:
        for i in range(len(out_fx)):
            out_fx[i]=(out_fx[i].T/meanflux).T
            out_fxt[i]=(out_fxt[i].T/meanflux).T
            out_err[i]=(out_err[i].T/meanflux).T
    return(out_wl,out_RV,out_fx,out_fxt,out_err,filelist,mjd,exptime,berv,orders_to_return,px_lims_to_return)




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



def read_linelists(textfile):
    """
    This reads a 2-column textfile that contains labels and paths to Kitzmann 2023 line lists.
    The first column contains a species label, the second contains the path to the datfile.


    Parameters
    ----------
    textfile : str, Path
        Path to a textfile that contains a 2-column textfile as NiI /path/to/Ni/list.dat.
        The first column is a label to use. The second is a path to a file that contains a 
        line list in 4 columns. Of this file, the second column is wavelength in um. 
        The fourth column is a measure of relative line strength.


    Returns
    -------
    labels : list
        List of species labels
    
    wl : list
        List of lists of wavelengths

    fx : list
        List of lists of line strengths.

    """   
    from data import test_exists

    test_exists(textfile)

    with open(textfile) as file:
        entries = [line for line in file]
    
    labels,wl,fx = [],[],[]
    for i in range(len(entries)):
        labels.append(entries[i].split()[0])

        A = np.loadtxt(entries[i].split()[1])
        wl.append(A[:,1]*1000.0)
        fx.append(A[:,3])

    return(labels,wl,fx)


def select_in_waveband(wlm,wlmin,wlmax):
    """
    From a list of wavelengths, select those wavelengths
    that fall between a minimum and a maximum.
    """
    wlm2 = []
    for w in wlm:
        if w>wlmin and w<wlmax:
            wlm2.append(w)
    return(np.array(wlm2))



def plot_labels(target_labels,labels,wlm,sm,dv=0,ax=None):
    """Wrapper for overplotting labels onto a spectrum."""
    gamma = (1.0+dv/3e5)

    if not ax:
        ax = plt.gca()
    for j in range(len(labels)):
        if labels[j] in target_labels:
            for i in range(len(wlm[j])):
                if sm[j][i]>0.0:
                    ax.plot(wlm[j][i]*np.array([1,1])*gamma,[0.0,1.1],color='black')
                    ax.text(wlm[j][i]*gamma,1.12,labels[j],color='black',ha='center')





