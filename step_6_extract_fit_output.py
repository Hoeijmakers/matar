from fitting import load_fit_output
import sys
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
import glob
import astropy.io.fits as fits
import pdb
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_6_extract_fit_output.py path/to/folder/containing/fitfiles.nc")
    
    inpath_1 = Path(str(sys.argv[1]))

filelist = []
for file in glob.glob(str(inpath_1/"*.nc")):
    if "out_samples" not in file:
        filelist.append(file)

mjds=[]
for i,f in enumerate(filelist):
    samples,dataslices,mjd = load_fit_output(f)

    if mjd == 0: #This is to deal with the possiblity that some fits were ran without metadata.
        f_stem = Path(f).stem
        hdr = fits.getheader(Path('/data/jens/observations/Betapic/HARPS2/'+str(f_stem)+'.fits'))
        mjds.append(hdr['MJD-OBS'])
    else: 
        mjds.append(mjd)

    if i == 0:
        params = list(samples.keys())
        nsamples = len(samples[params[0]])
        samples_all = np.zeros((len(filelist),len(params),nsamples))
    
    for j in range(len(params)):
        samples_all[i,j] = samples[params[j]]


sample_da = xr.DataArray(data=samples_all,dims=['time','params','samplenr'],coords=dict(time=mjds,params=params,samplenr=np.arange(nsamples)))

sample_da.to_netcdf(inpath_1/'out_samples.nc')










