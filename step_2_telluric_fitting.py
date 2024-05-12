from data import read_order,test_exists
import sys
import tayph.tellurics as tel
import tayph.system_parameters as sp
import tayph.operations as ops
from pathlib import Path
import numpy as np
import astropy.io.fits as fits
import pdb
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
if __name__ == "__main__":
    """ The strategy of this script is lifted essentially verbatim out of tayph.run.molecfit(). 
    There's a slight change of input, to the molecfit routines, but that is all."""


    if not len(sys.argv) > 3:
        raise Exception("Call as python3 step_2_telluric_fitting.py /input/directory/to/spectra.h5/ /path/to/configfile.dat /path/to/molecfit/config/file.dat")
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))
    molecfit_config = Path(str(sys.argv[3]))
    test_exists(inpath_1)
    test_exists(inpath_2)
    test_exists(molecfit_config)

    tel.test_molecfit_config(molecfit_config)
    molecfit_input_folder = Path(sp.paramget('molecfit_input_folder',molecfit_config,full_path=True))
    print(f'Molecfit input folder is {molecfit_input_folder}')
    molecfit_prog_folder = Path(sp.paramget('molecfit_prog_folder',molecfit_config,full_path=True))
    print(f'Molecfit executable is located at {molecfit_input_folder}')
    python_alias = sp.paramget('python_alias',molecfit_config,full_path=True)
    print(f'Python alias is {python_alias}')
    #If this passes, the molecfit confugration file appears to be set correctly.



    #Until now we have essentially followed the molecfit implementation of tayph. Here comes some of our own matar-specific input:
    C = np.loadtxt(inpath_2)
    if C[0] == 0:
        mode = 'gui'
    if C[0] == 1:
        mode = 'batch'
    if C[0] == 2:
        mode = 'both'

    if C[1] == 0:
        instrument = 'HARPS'
        norders = 72
        order_start = 39#Note that before this order, there are no meaningful tellurics and the molecfit continuum behaves weirdly.
    n = C[2]

    overwrite = bool(int(C[3]))



    wl_list,fx_list= [],[]
    wl,fx,err,filelist,mjd,exptime,berv = read_order(0,inpath_1)

    Nexp = len(filelist)
    npx = len(fx[0])





    #Now we can continue to run molecfit via tayph routines verbatim.
    #We check if the parameter file for this instrument exists.
    parname=Path(instrument+'.par')
    parfile = molecfit_input_folder/parname#Path to molecfit parameter file.
    test_exists(molecfit_input_folder)#This should be redundant, but still check.
    test_exists(parfile)#Test that the molecfit parameter file
    #for this instrument exists.


    mwl = 0.0
    if mode.lower() == 'gui' or mode.lower()=='both':
        for i in range(order_start,norders):
            wl,fx,void,filelist,mjd,exptime,berv = read_order(int(i),inpath_1,exp=int(n))
            fx_list.append(fx[wl>mwl])#This assumes that orders go from short wl to long wl.
            wl_list.append(wl[wl>mwl])
            mwl = np.max(wl) 

        wl_con = np.concatenate(wl_list)
        fx_con = np.concatenate(fx_list)
        print(f'Writing first spectrum {filelist[int(n)]} to {molecfit_input_folder} to start GUI')
        tel.write_file_to_molecfit(molecfit_input_folder,instrument+'.fits',[fits.getheader(filelist[int(n)])],[wl_con],[fx_con],0,plot=False)
        tel.execute_molecfit(molecfit_prog_folder,parfile,molecfit_input_folder,gui=True,alias=python_alias)
        tel.remove_output_molecfit(molecfit_input_folder,instrument)

    list_of_wlc,list_of_fxc,list_of_trans = [],[],[]
    if mode.lower() == 'batch' or mode.lower()=='both':
        print('Fitting %s spectra.' % len(filelist))
        for n in tqdm(range(len(filelist))):
            wl_list,fx_list,wl_list_complete,fx_list_complete = [],[],[],[] #Lists for cropped and full wavelength ranges.
            mwl = 0.0
            out_filename_split = str(filelist[int(n)].decode('utf-8')).split('.')
            out_filename_split[-2]+='_tel'
            out_filename = '.'.join(out_filename_split)
            # out_filename = out_filename.decode("utf-8")
            if overwrite == True or Path(out_filename).exists() == False:
                for i in range(order_start,norders):
                    wl,fx,void,void2,void3,void4,void5 = read_order(int(i),inpath_1,exp=int(n))
                    fx_list.append(fx[wl>mwl])
                    wl_list.append(wl[wl>mwl])
                    fx_list_complete.append(fx)
                    wl_list_complete.append(wl)
                    mwl = max(wl)



                wl_con = np.concatenate(wl_list)
                fx_con = np.concatenate(fx_list)
                wl_con_complete = np.concatenate(wl_list_complete)
                fx_con_complete = np.concatenate(fx_list_complete)
                tel.write_file_to_molecfit(molecfit_input_folder,instrument+'.fits',[fits.getheader(filelist[int(n)])],[wl_con],[fx_con],0)

                tel.execute_molecfit(molecfit_prog_folder,parfile,molecfit_input_folder,gui=False)
                print(f'Retrieving output from {str(molecfit_input_folder/instrument)}')
                wlc,fxc,trans = tel.retrieve_output_molecfit(molecfit_input_folder/instrument)
                #Note that regardless of the fact that we input air wavelengths, molecfit returns vaccuum wavelengths.
                #But we don't need to reverse engineer their airtovac routing because the output spectrum is matched to our input spectrum in air.
                # pdb.set_trace()
                trans_i = np.concatenate([np.arange(order_start*npx)*0.0+1.0,interp1d(wl_con,trans)(wl_con_complete)])
                #Most of the values in wlc and wl_con_complete are identical, and so are the interpolates.
                #This is only to evaluate what's in the overlap regions.
                #And expanding with ones all the wavelengths that we didn't bother to execute.

                fits.writeto(out_filename,np.reshape(trans_i,(norders,npx)),overwrite=True,header=fits.getheader(filelist[int(n)].decode('utf-8')),output_verify='fix+warn')
                tel.remove_output_molecfit(molecfit_input_folder,instrument)
            else:
                print(f'Skipping {out_filename}: Already exists.')