from data import test_exists
from analysis import load_order_RV_range
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
from matplotlib.backends.backend_pdf import PdfPages
import pdb
from tqdm import tqdm
import astropy.constants as const
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_4_plot_slice.py path/to/spectra.h5 center_wl RV_range vsys ymax")
    
    inpath_1 = Path(str(sys.argv[1]))
    wlc = float(sys.argv[2])
    drv = float(sys.argv[3])

    if len(sys.argv)>4:
        vsys= float(sys.argv[4])
    else:
        vsys = 0.0

    if len(sys.argv)>5:
        ymax = float(sys.argv[5])
    else:
        ymax = 3.0


    c = const.c.to('km/s').value

    test_exists(inpath_1)
    #wl,RV,spec_norm,R,mean_clean_spec,mjd,t_exp = read_order(inpath_1)
    # selected_wl,selected_data,np.sqrt(selected_data),filelist,mjd,exptime,berv
    # wl,spec,err,filelist,mjd,t_exp = read_order(inpath_1)

    wl,RV,orders_norm,err,filelist,mjd,t_exp,berv,orders_returned,px_lims_returned = load_order_RV_range(inpath_1,wlc*(1+vsys/c),drv+33,bervcor=True)#33 km/s is added because we will still be BERV correcting and 33km/s is the maximum possible BERV.
    #Note that wl is now berv-corrected and fx is mean-normalised.

    # for i in range(20):
    #     for j in range(len(orders_returned)):
    #         plt.plot(wl[j][i],orders_norm[j][i],color=f'C{j}',linewidth=0.7,alpha=0.2)
    # plt.show()

  

    mjd_boundary = 0.65 #This is typical daytime.

    mjd_start = math.floor(min(mjd))-1
    mjd_end = math.ceil(max(mjd)+1)

    N,T,S,D,L,RVS = [],[],[],[],[],[]
    for d in range(mjd_start,mjd_end,1):
        l1 = d+mjd_boundary
        l2 = d+mjd_boundary+1
        sel = (mjd<l2)&(mjd>l1)
        if len(mjd[sel]) > 0:
            N.append((len(mjd[sel])))
            T.append(np.sum(t_exp[sel]))
            D.append(min(mjd[sel]))
            L.append(l1)
            
            S_orders,RV_orders = [],[]

            for order in orders_norm:
                S_day = []
                for s in order[sel]:
                    S_day.append(s)
                S_orders.append(S_day)
            S.append(S_orders)

            for v_axis in RV:
                RV_day = []
                for r in v_axis[sel]:
                    RV_day.append(r)
                RV_orders.append(RV_day)
            RVS.append(RV_orders)

    
    from astropy.time import Time

    N_p_max = 8
    N_panels = len(N)#N is the number of spectra in each night, its length is the number of nights.

    # ysel = (RV>-RVr)&(RV<RVr)
    # ylim = [0,np.max(mean_clean_spec[ysel])]
    ylim = [0,ymax]
    with PdfPages('multipage_pdf.pdf') as pdf:
        for i in tqdm(range(N_panels)):
            if i%N_p_max == 0:
                fig,ax = plt.subplots(N_p_max,1,figsize=(10,15),sharex=True)

            if len(S[i][0]) == 1:
                col = plt.cm.plasma([0.0]) 
            else:
                col = plt.cm.plasma(np.linspace(0,1,len(S[i][0])))

            # pdb.set_trace()
            for j in range(len(S[0])):
                for ii,s in enumerate(S[i][j]):
                    try:
                        ax[i%N_p_max].plot(RVS[i][j][ii],S[i][j][ii],color=col[ii],alpha=np.max(np.array([0.1,1/np.sqrt(N[i])])))
                    except:
                        pdb.set_trace()
            datetime=Time(D[i],format='mjd').fits.replace('T',' ')
            ax[i%N_p_max].set_xlim(-drv,drv)
            ax[i%N_p_max].set_ylim(ylim)
            ax[i%N_p_max].text(drv-4,0.4*ylim[1],f'Date: {datetime}',ha='right')
            ax[i%N_p_max].text(drv-4,0.3*ylim[1],f'N: {len(S[i][0])}',ha='right')
            ax[i%N_p_max].text(drv-4,0.2*ylim[1],f't: {np.round(T[i]/3600,2)}h',ha='right')
            ax[i%N_p_max].text(drv-4,0.1*ylim[1],f'mjd: {L[i]} - {L[i]+1}',ha='right')
            if i%N_p_max == N_p_max-1:  
                fig.tight_layout()          
                pdf.savefig(fig)  # saves the current figure into a pdf page
                plt.close()


        fig.tight_layout()          
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close()

