import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import tayph.util as ut
from data import test_exists,read_slice,construct_dataframe
from processing import normslice,line_core_fit
from analysis import select_high_spectra
import pdb

if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 main.py /input/directory/to/fits/files/ optional_filename.h5 optional_out_filename")
    datafolder = Path(str(sys.argv[1]))
    test_exists(datafolder)
    if len(sys.argv) > 2:
        inpath = datafolder/str(sys.argv[2])
    else:
        inpath = datafolder/'spectra.h5'
    test_exists(inpath)

    if len(sys.argv) > 3:
        outpath = Path(str(sys.argv[3])+'.txt')
        outpath2= Path(str(sys.argv[3])+'.h5')
    else:
        outpath = datafolder/'selected_spectra.txt'
        outpath2 = datafolder/'selected_spectra.h5'


    #This reads the data saved in the dataframe. For small slices it does so very quickly.
    t1=ut.start()
    wl,spec,filename,mjd,exptime=read_slice(392.21,398.5,inpath)
    spec_norm = normslice(wl,spec,deg=3,reject_regions=[[393.1,393.7],[396.5,397.2]],plot=True)
    ut.end(t1)

  


    #We then fit the line core, excluding the narrow range around the line center.
    wl_narrow,spec_narrow,R,fit=line_core_fit(wl,spec_norm,range_include=[393.23,393.58],range_ignore=[393.374,393.423],deg=5,plot=False)
    #This is done after the data have been made into a dataframe, and after removing from the input folder some spectra 
    #that may be bad, for which the inspect_spectra function has been written.


    #We plot the residuals, and select ranges within which to test the flux.
    plt.pcolormesh(wl_narrow,np.arange(len(R)),R,vmin=0.9,vmax=1.05)
    wlc = [393.295,393.344,393.358,393.369,393.379,393.410,393.425,393.445,393.4618,393.4847,393.52]
    dwl = [0.01,0.005,0.005,0.005,0.003,0.005,0.005,0.005,0.005,0.01,0.015]
    for i in range(len(wlc)):
        plt.axvspan(wlc[i]-dwl[i],wlc[i]+dwl[i],color='red',alpha=0.2)
    plt.show()


    #We then select all spectra from the series where the flux is high in these bands.
    nominal_spec_i = select_high_spectra(wl_narrow,R,wlc,dwl,threshold=0.925,plot=False)

    #We determine the mean spectrum of those and plot it.
    mean_clean_spec = np.nanmedian(spec_norm[nominal_spec_i],axis=0)
    for i in nominal_spec_i:
        plt.plot(wl,spec_norm[i],color='black',alpha=0.05)
    plt.plot(wl,mean_clean_spec,color='red',label='Median spectrum')
    plt.legend()
    plt.show()


    #We compute the residuals. This is our 'clean' data.
    # R2 = spec_norm/mean_clean_spec
    # plt.pcolormesh(wl,np.arange(len(R2)),R2,vmin=0.9,vmax=1.05,cmap='copper')
    # plt.title('Residuals after removing median of selected clean spectra')
    # plt.show()



    #We need to do a second pass because some absorbing spectra remain.
    R3 = spec_norm[nominal_spec_i]/mean_clean_spec
    wlc = [393.276,393.435,393.4823,393.550]
    dwl = [0.015,0.01,0.015,0.015]
    plt.pcolormesh(wl,np.arange(len(R3)),R3,vmin=0.95,vmax=1.025)
    for i in range(len(wlc)):
        plt.axvspan(wlc[i]-dwl[i],wlc[i]+dwl[i],color='red',alpha=0.2)
    plt.title("Residual of 'clean' spectra.")
    plt.xlim(393.176, 393.731)
    plt.show()


    #Get the indices of the indices of the cleanest ones among the clean spectra.
    nominal_spec_i_i = select_high_spectra(wl,R3,wlc,dwl,threshold=0.97,plot=True)


    # c=list(np.array(a)[b].astype(int))
    nominal_spec_i_2=list(np.array(nominal_spec_i)[nominal_spec_i_i].astype(int))

    #Update the mean spectrum and recompute the residuals.
    mean_clean_spec_2 = np.nanmedian(spec_norm[nominal_spec_i_2],axis=0)


    wlc = [393.226,393.55]
    dwl = [0.015,0.015]
    R4 = spec_norm[nominal_spec_i_2]/mean_clean_spec_2
    plt.pcolormesh(wl,np.arange(len(R4)),R4,vmin=0.95,vmax=1.025)
    for i in range(len(wlc)):
        plt.axvspan(wlc[i]-dwl[i],wlc[i]+dwl[i],color='red',alpha=0.2)
    plt.title("Residual of 'clean' spectra version 2.")
    plt.xlim(393.176, 393.731)
    plt.show()
    
    
    #Still we are not happy so we do it again.
    nominal_spec_i_2_i = select_high_spectra(wl,R4,wlc,dwl,threshold=0.98,plot=True)
    nominal_spec_i_3=list(np.array(nominal_spec_i_2)[nominal_spec_i_2_i].astype(int))
    mean_clean_spec_3 = np.nanmedian(spec_norm[nominal_spec_i_3],axis=0)


    wlc = [393.310,393.42,393.520,393.585,393.63]
    dwl = [0.015,0.006,0.015,0.015,0.015]
    R5 = spec_norm[nominal_spec_i_3]/mean_clean_spec_3
    plt.pcolormesh(wl,np.arange(len(R5)),R5,vmin=0.95,vmax=1.025)
    for i in range(len(wlc)):
        plt.axvspan(wlc[i]-dwl[i],wlc[i]+dwl[i],color='red',alpha=0.2)
    plt.title("Residual of 'clean' spectra version 3.")
    plt.xlim(393.176, 393.731)
    plt.show()  

    #Still we are not happy so we do it again.
    nominal_spec_i_3_i = select_high_spectra(wl,R5,wlc,dwl,threshold=0.98,plot=True)
    nominal_spec_i_4=list(np.array(nominal_spec_i_3)[nominal_spec_i_3_i].astype(int))
    mean_clean_spec_4 = np.nanmedian(spec_norm[nominal_spec_i_4],axis=0)
    R6 = spec_norm[nominal_spec_i_4]/mean_clean_spec_4
    plt.pcolormesh(wl,np.arange(len(R6)),R6,vmin=0.95,vmax=1.025)
    plt.title("Residual of 'clean' spectra version 4.")
    plt.xlim(393.176, 393.731)
    plt.show()  



    for i in range(len(nominal_spec_i)):
        plt.plot(wl,spec_norm[nominal_spec_i[i]],alpha=0.15,color='black',linewidth=0.6)
    for i in range(len(nominal_spec_i_4)):
        plt.plot(wl,spec_norm[nominal_spec_i_4[i]],alpha=0.2,color='red',linewidth=0.6)
    plt.title(f'Collection of {int(len(nominal_spec_i_4))} clean spectra (first pass vs last pass).')
    plt.plot(wl,mean_clean_spec_4,linewidth=2,color='white')
    plt.show()


    #We compute the residuals. This is our 'clean' data.
    R7 = spec_norm/mean_clean_spec_4
    plt.pcolormesh(wl,np.arange(len(R7)),R7,vmin=0.9,vmax=1.05,cmap='copper')
    plt.title(f'Residuals after removing median of {int(len(nominal_spec_i_4))} selected clean spectra')
    plt.show()

    plt.plot(wl,np.nanmean(R7,axis=0))
    plt.title('Mean residual spectrum')
    plt.show()

    with open(outpath,'w') as fp:
        for item in filename[nominal_spec_i_4]:
            fp.write(f"{item.decode('unicode_escape')}\n")
    print(f'Assigned {int(len(nominal_spec_i_4))} files in {str(outpath)}')

    construct_dataframe(outpath,outpath=outpath2) 





