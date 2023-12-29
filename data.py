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
    if not inpath.exists():
        raise Exception(f"{str(inpath)} not found.")

def read_spectrum(f):
    """
        This reads a HARPS ADP spectrum. ADP spectra are BERV-corrected so they are in a constant
        stellar frame. Wavelength is returned in nm.
    """

    from pathlib import Path
    import astropy.io.fits as fits
    import numpy as np

    file = Path(f)
    test_exists(file)

    with fits.open(file) as hdul:
        spec=hdul[1].data[0]

    return(spec[0]/10.0,spec[1],spec[2])


def construct_df(list_of_files,outpath):
    """
    This takes a list of filepaths to ADP fits files and saves them as a pandas dataframe in an h5
    file, interpolating the spectra to one wavelength grid.
    This file is to be read by the read_slice function below.
    """
    import pandas as pd
    import numpy as np
    import scipy.interpolate as interp
    from tqdm import tqdm
    import h5py
    wl_base,void1,void2 = read_spectrum(list_of_files[0])
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
        file['wavelength'][:]=wl_base
        # Simulate appending rows iteratively
        for i in tqdm(range(len(list_of_files))):
            wl,fx,err = read_spectrum(list_of_files[i])
            fx_i = interp.interp1d(wl,fx,bounds_error=False,fill_value=0)(wl_base)
            err_i = interp.interp1d(wl,err,bounds_error=False,fill_value=0)(wl_base)#This is
            #an approximation, assuming that the error of adjacent points is nearly identical.
            #In reality, an error propagation needs to be carried out.
            current_size = file['spectra'].shape[0]
            # Resize the dataset to accommodate the new row
            file['spectra'].resize((current_size + 1, len(wl_base)))
            # Append the new row to the dataset
            file['spectra'][current_size, :] = fx_i



def read_slice(min_wavelength,max_wavelength,hdf_file_path):
    # Set your minimum and maximum wavelength values
    import h5py
    import pdb

    # Check if the dataset exists in the HDF5 file
    with h5py.File(hdf_file_path, 'r') as file:
        wl = np.array(file['wavelength'])
        filelist = np.array(file['filenames'])
        indices = np.arange(len(wl))[(wl >= min_wavelength) & (wl <= max_wavelength)]
        selected_data = np.array(file['spectra'][:, min(indices):max(indices)+1])
        selected_wl = np.array(file['wavelength'][min(indices):max(indices)+1])
        return(selected_wl,selected_data,filelist)





if __name__ == "__main__":
    # This code will only run if the script is executed directly

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 main.py /input/directory/to/fits/files/")
    inpath = Path(str(sys.argv[1]))

    test_exists(inpath)

    search_string = str(inpath/'*.fits')
    filelist = glob.glob(search_string)

    if not len(filelist)>0:
        raise Exception(f"No FITS files encountered in {inpath}")

    if len(sys.argv) > 2:
        N = int(sys.argv[2])
        filelist=filelist[0:N]
    construct_df(filelist,inpath/'spectra.h5')



    #
    # print(np.shape(spec))
    # pdb.set_trace()
    # plt.imshow((spec.T/np.nanmean(spec,axis=1)).T,aspect='auto')
    # plt.show()
