from data import read_fits,test_exists
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
from matplotlib.backends.backend_pdf import PdfPages
import pdb
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_10_viewer.py /input/directory/to/fits/slice.fits")
    
    inpath_1 = Path(str(sys.argv[1]))


    test_exists(inpath_1)

    wl,RV,spec_norm,R,mean_clean_spec,mjd,t_exp = read_fits(inpath_1)


    mjd_boundary = 0.65 #This is typical daytime.

    mjd_start = math.floor(min(mjd))-1
    mjd_end = math.ceil(max(mjd)+1)

    N,T,S,D = [],[],[],[]
    for d in range(mjd_start,mjd_end,1):
        l1 = d+mjd_boundary
        l2 = d+mjd_boundary+1
        sel = (mjd<l2)&(mjd>l1)
        if len(mjd[sel]) > 0:
            N.append((len(mjd[sel])))
            T.append(np.sum(t_exp[sel]))
            D.append(min(mjd[sel]))
            S_day = []
            for s in spec_norm[sel]:
                S_day.append(s)
            S.append(S_day)

    
    from astropy.time import Time

    N_p_max = 8
    N_panels = len(N)
    RVr = 60.0
    ysel = (RV>-RVr)&(RV<RVr)
    ylim = [0,np.max(mean_clean_spec[ysel])]
    with PdfPages('multipage_pdf.pdf') as pdf:
        for i in range(len(N)):
            if i%N_p_max == 0:
                fig,ax = plt.subplots(N_p_max,1,figsize=(10,15),sharex=True)

            if len(S[i]) == 1:
                col = plt.cm.plasma([0.0]) 
            else:
                col = plt.cm.plasma(np.linspace(0,1,len(S[i])))
            for ii,s in enumerate(S[i]):
                ax[i%N_p_max].plot(RV,s,color=col[ii],alpha=np.max([0.1,1/np.sqrt(N[i])]))
            datetime=Time(D[i],format='mjd').fits.replace('T',' ')
            ax[i%N_p_max].set_xlim(-RVr,RVr)
            ax[i%N_p_max].set_ylim(ylim)
            ax[i%N_p_max].text(RVr-4,0.3*ylim[1],f'Exp 1: {datetime}',ha='right')
            ax[i%N_p_max].text(RVr-4,0.2*ylim[1],f'N: {len(S[i])}',ha='right')
            ax[i%N_p_max].text(RVr-4,0.1*ylim[1],f't: {np.round(T[i]/3600,2)}h',ha='right')
            if i%N_p_max == N_p_max-1:  
                fig.tight_layout()          
                pdf.savefig(fig)  # saves the current figure into a pdf page
                plt.close()

   



    # plt.hist(mjd%1)
    # plt.show()