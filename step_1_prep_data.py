from pathlib import Path
import sys
import glob
from data import construct_df,test_exists

if __name__ == "__main__":

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 data.py /input/directory/to/fits/files/ 200 optional_out_filename.h5")
    datafolder = Path(str(sys.argv[1]))
    test_exists(datafolder)
    if len(sys.argv) > 3:
        outpath = datafolder/str(sys.argv[3])
    else:
        outpath = datafolder/'spectra.h5'


    search_string = str(datafolder/'*.fits')
    filelist = glob.glob(search_string)

    if not len(filelist)>0:
        raise Exception(f"No FITS files encountered in {datafolder}")

    if len(sys.argv) > 2:
        N = int(sys.argv[2])
        if N > 0:
            filelist=filelist[0:N]
    construct_df(filelist,outpath) 