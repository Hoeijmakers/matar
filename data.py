import os
import numpy as np
from pathlib import Path
import astropy.io.fits as fits
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import tayph.functions as fun
import tayph.operations as ops
import sys
import glob
import pdb
import pandas as pd
import tayph.util as ut


def test_exists(inpath):
    """
    This tests if a file exists.
    """
    if not inpath.exists():
        raise Exception(f"{str(inpath)} not found.")

def read_spectrum(f):
    """
        This reads a HARPS ADP spectrum. ADP spectra are BERV-corrected so they are in a constant
        stellar inertial frame. Wavelength is returned in nm.
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


def construct_df(list_of_files,outpath):
    """
    This takes a list of filepaths to ADP fits files and saves them as an h5 datastructure
    interpolating the spectra to one wavelength grid. This file is to be read by the 
    read_slice function below.
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



def read_slice(min_wavelength,max_wavelength,hdf_file_path):
    """This reads a slice out of an h5 datastructure written by construct_df above.
    It only reads between min_wavelength and max_wavelength, so you can deal with a large
    multitude of ADP spectra without loading them all in memory.
    """

    # Set your minimum and maximum wavelength values
    import h5py
    import pdb

    # Check if the dataset exists in the HDF5 file
    with h5py.File(hdf_file_path, 'r') as file:
        wl = np.array(file['wavelength'])
        filelist = np.array(file['filenames'])
        mjd = np.array(file['mjd'])
        indices = np.arange(len(wl))[(wl >= min_wavelength) & (wl <= max_wavelength)]
        selected_data = np.array(file['spectra'][:, min(indices):max(indices)+1])
        selected_wl = np.array(file['wavelength'][min(indices):max(indices)+1])
        return(selected_wl,selected_data,filelist,mjd)


def construct_dataframe(inpath,N=0,outpath=''):
    """This constructs an h5 datafile from a list of fits files co-located in a folder.
    Set N to an integer to include only the first N spectra (to reduce data volume).
    The outpath is set optionally. If not set, the outpath is the same as the inpath, 
    creating a file called spectra.h5.
    The inpath can also be set to a textfile that contains a list of filepaths.
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

    if len(outpath) > 0:
        construct_df(filelist,outpath)       
    else:
        construct_df(filelist,inpath/'spectra.h5')
    return