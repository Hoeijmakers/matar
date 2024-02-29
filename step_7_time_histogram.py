from data import read_slice,test_exists
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_6_measure_absorption.py /input/directory/to/fits/files.fits")
    
    inpath_1 = Path(str(sys.argv[1]))
    inpath_2 = Path(str(sys.argv[2]))

    test_exists(inpath_1)
    test_exists(inpath_2)

    wl,spec_slice,filename,mjd,exptime=read_slice(500,500.42,inpath_1)
    wl2,spec_slice2,filename2,mjd2,exptime2=read_slice(500,500.42,inpath_2)


    mjd_min = math.floor(np.min(mjd))-1#This is the date-limit, the daytime.
    mjd_max = math.floor(np.max(mjd))+1

    mjd = np.array(mjd)
    exptime = np.array(exptime)
    time_totals = []
    N_totals = []
    bin_edges = []
    dt = 1

    from astropy.time import Time
  
    for i in range(int(mjd_min),int(mjd_max),dt):
        total_time = np.sum(exptime[(mjd>=(i-0.25))&(mjd<(i+dt-0.25))])
        total_N = len(exptime[(mjd>=(i-0.25))&(mjd<(i+dt-0.25))])
        if total_N>0:
            bin_edges.append(i-0.25)
            time_totals.append(total_time)
            N_totals.append(total_N)

    t = Time(bin_edges, format='mjd', scale='utc')
    years = t.decimalyear

    n_panels = math.floor(np.max(years))-math.floor(np.min(years))

    fig,ax = plt.subplots(n_panels,1,figsize=(10,10))
    plt.subplots_adjust(left=0.025,
                    bottom=0.03, 
                    right=0.99, 
                    top=0.99, 
                    wspace=0.1, 
                    hspace=0.06)
    for i in range(n_panels):
        ax[i].bar(bin_edges,time_totals,align='edge',color='navy',alpha=0.8,width=0.9)
        tmin = Time(i+math.floor(np.min(years)),format='decimalyear',scale='utc').mjd
        tmax = Time(1+i+math.floor(np.min(years)),format='decimalyear',scale='utc').mjd
        ax[i].set_xlim(tmin,tmax)
        ax[i].set_yscale('log')
        ax[i].text(0.01, 0.8,f'{i+math.floor(np.min(years))}',horizontalalignment='left',verticalalignment='center',transform = ax[i].transAxes)
        ax[i].tick_params(axis='both', which='major', labelsize=6)
        ax[i].axhline(3600,color='red',linewidth=0.6,alpha=0.8)
        # ax[i].tick_params(axis='both', which='minor', labelsize=9)
        
    # plt.bar(bin_edges,time_totals,align='edge',color='navy',alpha=0.8,width=0.9)
    # fig.tight_layout()
    plt.show()
