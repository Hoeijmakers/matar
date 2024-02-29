from data import read_fits,test_exists
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_6_measure_absorption.py /input/directory/to/fits/files.fits")
    
    inpath_1 = Path(str(sys.argv[1]))


    RVmin = 15
    RVmax = 35


    wl,RV,spec_norm,R,mean_clean_spec,mjd = read_fits(inpath_1)
    if len(sys.argv) > 2:
        inpath_2 = Path(str(sys.argv[2]))   
        wl2,RV2,spec_norm2,R2,mean_clean_spec2,mjd2 = read_fits(inpath_2)           
    # plt.pcolormesh(RV,np.arange(len(R)),R,vmin=0.8,vmax=1.05)
    # plt.show()

    plt.plot(RV2,np.nanmean(R2,axis=0))
    plt.show()


    abs_1_blue = np.nanmean(R[:,(RV>-RVmax)&(RV<-RVmin)],axis=1)
    abs_1_red = np.nanmean(R[:,(RV>RVmin)&(RV<RVmax)],axis=1)
    if len(sys.argv) > 2:
        abs_2_blue = np.nanmean(R2[:,(RV2>-RVmax)&(RV2<-RVmin)],axis=1)
        abs_2_red = np.nanmean(R2[:,(RV2>RVmin)&(RV2<RVmax)],axis=1)

        
    if len(sys.argv) == 2:
        plt.plot(mjd,abs_1_blue,'.')
        plt.plot(mjd,abs_1_red,'.')
        plt.show()
    if len(sys.argv) > 2:
        fig,ax = plt.subplots(2,2,sharey=True,sharex=True)
        ax[0][0].set_title(f'Absorption {inpath_1} from $\pm${RVmin} - $\pm${RVmax}')
        ax[0][0].plot(mjd,abs_1_blue,'.',label='Blue-shifted')
        ax[0][0].plot(mjd,abs_1_red,'.',label='Red-shifted')
        ax[0][1].set_title(f'Absorption {inpath_2} from $\pm${RVmin} - $\pm${RVmax}')
        ax[0][1].plot(mjd,abs_2_blue,'.',label='Blue-shifted')
        ax[0][1].plot(mjd,abs_2_red,'.',label='Red-shifted')

        ax[1][0].set_title(f'Compare {inpath_1} and {inpath_2} blue-shifted')
        ax[1][0].plot(mjd,abs_1_blue,'.',label='Blue-shifted 1st')
        ax[1][0].plot(mjd,abs_2_blue,'.',label='Blue-shifted 2nd')
        ax[1][1].set_title(f'Compare {inpath_1} and {inpath_2} red-shifted')
        ax[1][1].plot(mjd,abs_1_red,'.',label='Red-shifted 1st')
        ax[1][1].plot(mjd,abs_2_red,'.',label='Red-shifted 2nd')
        plt.show()


    if len(sys.argv) > 2:
        plt.plot(1-abs_1_red,1-abs_2_red,'.',alpha=0.2,ms=3,label='Red-shifted lines',color='indianred')
        plt.plot(1-abs_1_blue,1-abs_2_blue,'.',alpha=0.2,ms=3,label='Blue-shifted lines',color='cornflowerblue')
        plt.xlabel(f'Absorption {inpath_1}')
        plt.ylabel(f'Absorption {inpath_2}')
        ax = plt.gca()
        lim1 = min(ax.get_ylim()+ax.get_xlim())
        lim2 = max(ax.get_ylim()+ax.get_xlim())
        plt.xlim(lim1,lim2)
        plt.ylim(lim1,lim2)
        plt.plot(np.linspace(lim1,lim2,10),np.linspace(lim1,lim2,10),linewidth=2,color='black',label='Optically thick limit')
        plt.plot(2*np.linspace(lim1,lim2,10),np.linspace(lim1,lim2,10),color='black',label='Optically thin limit')
        plt.legend()
        plt.show()