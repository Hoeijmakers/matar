import os
import numpy as np
from pathlib import Path
import glob


def test_exists(inpath):
    """
    This tests if a file exists.

    Parameters
    ----------
    inpath : str, Path
        Filepath to be tested.
    """
    if not Path(inpath).exists():
        raise Exception(f"{str(inpath)} not found.")

def read_spectrum(f):
    """
        This reads a HARPS ADP spectrum. ADP spectra are BERV-corrected so they are in a constant
        stellar inertial frame. Wavelength is returned in nm.

    Parameters
    ----------
    f : str, Path
        Path to a HARPS ADP FITS file.

    Returns
    -------
    wl : array
        Wavelength axis in nm.
    
    fx : array
        Flux axis.

    err : array
        Uncertainties axis.
    
    hdr : FITS header object
        Header.
    """

    from pathlib import Path
    import astropy.io.fits as fits
    import numpy as np

    file = Path(f)
    test_exists(file)

    with fits.open(file) as hdul:
        spec=hdul[1].data[0]
        hdr=hdul[0].header

    return(spec[0]/10.0,spec[1],spec[2],hdr)


def test_wl_grid(inpath):
    """
    This tests the integrity of the wavelength axes of a list of filepaths to HARPS ADP fits files.
    This is done prior to saving them to a datastructure with construct_dataframe, which assumes that
    all wavelength axes are identical.


    Parameters
    ----------

    list_of_files : list, tuple
        List of filenames associated with a spectroscopic time-series to be stored.
        The structure has the following dict keys: wavelength, spectra, filenames, mjd.

    """
    import os
    from pathlib import Path
    import pdb
    from tqdm import tqdm

    inpath = Path(str(inpath))

    if os.path.isdir(inpath):
        test_exists(inpath)
        search_string = str(inpath/'*.fits')
        filelist = glob.glob(search_string)
    else:
        with open(inpath) as file:
            filelist = [line.rstrip() for line in file]
        for file in filelist:
            test_exists(file)

    if not len(filelist)>0:
        raise Exception(f"No FITS files encountered in {inpath}")

    wl_list = []
    for i in tqdm(range(len(filelist))):
        wl,fx,err,hdr = read_spectrum(filelist[i])

        wl_list.append(wl)

    for i in range(0,9000,900): 
        print(min(wl_list[i]))




def construct_df(list_of_files,outpath):
    """
    This takes a list of filepaths to HARPS ADP fits files and saves them as an h5 datastructure
    interpolating the spectra to one wavelength grid. It is assumed that all files are defined on
    the same wavelength grid.

    The output file is meant to be read by the read_slice function below.

    Parameters
    ----------

    list_of_files : list, tuple
        List of filenames associated with a spectroscopic time-series to be stored.
        The structure has the following dict keys: wavelength, spectra, filenames, mjd.


    outpath : str, Path
        The output h5 file.

    """
    import numpy as np
    import scipy.interpolate as interp
    from tqdm import tqdm
    import h5py
    wl_base,void1,void2,void3 = read_spectrum(list_of_files[0])
    del void1
    del void2
    if os.path.exists(outpath):
        os.remove(outpath)


    # Create or open the HDF5 file in append mode
    with h5py.File(outpath, 'a') as file:
        # Create a dataset 'your_dataset_name' if it doesn't exist
        if 'spectra' not in file:
            # Assuming your data has a shape of (number_of_rows, number_of_columns)
            dataset_shape = (0, len(wl_base))  # Update the number_of_columns accordingly
            maxshape = (None, len(wl_base))  # Update the number_of_columns accordingly
            dtype = np.float64  # Adjust the dtype as per your data type

            # Create the dataset with an extendable shape
            dataset = file.create_dataset('spectra', shape=dataset_shape, maxshape=maxshape, dtype=dtype)

        if 'filenames' not in file:
            filenames=file.create_dataset('filenames',shape=(len(list_of_files)),dtype=h5py.special_dtype(vlen=str))
        file['filenames'][:]=list_of_files
        if 'wavelength' not in file:
            wavelength=file.create_dataset('wavelength',shape=(len(wl_base)),dtype=dtype)
        if 'mjd' not in file:
            mjd=file.create_dataset('mjd',shape=(len(list_of_files)),dtype=dtype)  
        if 'exptime' not in file:
            mjd=file.create_dataset('exptime',shape=(len(list_of_files)),dtype=dtype)  
        file['wavelength'][:]=wl_base

        # Simulate appending rows iteratively
        for i in tqdm(range(len(list_of_files))):
            wl,fx,err,hdr = read_spectrum(list_of_files[i])
            fx_i = interp.interp1d(wl,fx,bounds_error=False,fill_value=0)(wl_base)
            err_i = interp.interp1d(wl,err,bounds_error=False,fill_value=0)(wl_base)#This is
            #an approximation, assuming that the error of adjacent points is nearly identical.
            #In reality, an error propagation needs to be carried out.
            current_size = file['spectra'].shape[0]
            # Resize the dataset to accommodate the new row
            file['spectra'].resize((current_size + 1, len(wl_base)))
            # Append the new row to the dataset
            file['spectra'][current_size, :] = fx_i
            file['mjd'][i]=hdr['MJD-OBS']
            file['exptime'][i]=hdr['EXPTIME']



def read_slice(min_wavelength,max_wavelength,hdf_file_path):
    """This reads a slice out of an h5 datastructure written by construct_df above.
    It only reads between min_wavelength and max_wavelength, so you can deal with a large
    multitude of ADP spectra without loading them all in memory.


    Parameters
    ----------
    min_wavelength : float
        1D array of the input wavelength range, matching the width of spec_norm.

    max_wavelength : float
        2D array of spectroscopic time-series or residuals. 
        Its width matches the length of wl. 

    hdf_file_path : list, tuple
        Path to an h5 file generated by construct_df above.

    cutoff : float, optional
        The limiting standard deviation over which a spectrum is selected.
    
    alpha : float, optional
        The plotting transparency. Set to a small value if selecting many spectra at once.

    Returns
    -------
    selected_wl : array
        Selected part of the wavelength axis.

    selected_data : array
        2D array of the spectra between the wavelength limits.

    filelist : list
        list of the files in this dataframe.

    mjd : list
        list of timestamps associated with this data, in mjd.

    """

    import h5py

    # Check if the dataset exists in the HDF5 file
    with h5py.File(hdf_file_path, 'r') as file:
        wl = np.array(file['wavelength'])
        filelist = np.array(file['filenames'])
        mjd = np.array(file['mjd'])
        exptime = np.array(file['exptime'])
        indices = np.arange(len(wl))[(wl >= min_wavelength) & (wl <= max_wavelength)]
        selected_data = np.array(file['spectra'][:, min(indices):max(indices)+1])
        selected_wl = np.array(file['wavelength'][min(indices):max(indices)+1])
    return(selected_wl,selected_data,filelist,mjd,exptime)


def construct_dataframe(inpath,N=0,outpath=''):
    """This constructs an h5 datafile from a list of fits files co-located in a folder.
    Set N to an integer to include only the first N spectra (to reduce data volume).
    The outpath is set optionally. If not set, the outpath is the same as the inpath, 
    creating a file called spectra.h5.
    The inpath can also be set to a textfile that contains a list of filepaths.
    It wraps around the construct_df function above.


    Parameters
    ----------
    inpath : str, Path
        Path to a datafolder to be used as input, containing a list of FITS files, or
        a path to a textfile that contains a list of filenames.

    N : int
        The number of spectra to take into account from the list.

    outpath : str, Path
        Path to the output h5 file.


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
    import os
    from pathlib import Path

    inpath = Path(str(inpath))
    test_exists(inpath)

    if os.path.isdir(inpath):
        search_string = str(inpath/'*.fits')
        filelist = glob.glob(search_string)
    else:
        with open(inpath) as file:
            filelist = [line.rstrip() for line in file]
        for file in filelist:
            test_exists(file)

    
    if not len(filelist)>0:
        raise Exception(f"No FITS files encountered in {inpath}")
    if N>0:
        filelist=filelist[0:N]

    # test_wl_grid(filelist)
    if len(str(outpath)) > 0:
        construct_df(filelist,outpath)       
    else:
        construct_df(filelist,inpath/'spectra.h5')


def read_fits(filename):
    """This reads a fits datastructure as saved in step 4"""
    import astropy.io.fits as fits

    test_exists(filename)
    with fits.open(filename) as hdul:
        spec_norm = hdul[0].data
        wl = hdul[1].data
        R = hdul[2].data
        mean_clean_spec = hdul[3].data
        RV = hdul[4].data
        mjd = hdul[5].data
    return(wl,RV,spec_norm,R,mean_clean_spec,mjd)

