from pathlib import Path
import sys
import glob
from data import construct_df2d,test_exists

if __name__ == "__main__":

    if not len(sys.argv) > 1:
        raise Exception("Call as python3 step_1_prep_data.py /input/directory/to/fits/files/ optional_out_filename.h5 0 100")
    datafolder = Path(str(sys.argv[1]))
    test_exists(datafolder)
    if len(sys.argv) > 2:
        outpath = datafolder/str(sys.argv[2])
    else:
        outpath = datafolder/'spectra.h5'


    search_string = str(datafolder/'*_e2ds_A.fits')
    filelist = glob.glob(search_string)

    if not len(filelist)>0:
        raise Exception(f"No e2ds FITS files encountered in {datafolder}")
    else:
        print(f'{len(filelist)} found in {str(datafolder)}')
    if len(sys.argv) > 3:
        Nstart = int(sys.argv[3])
        Nend   = int(sys.argv[4])
        print(f'Limiting data read to {Nstart} - {Nend}')
        if Nstart < 0  or Nstart > len(filelist):
            raise Exception(f'Nstart ({Nstart}) should be smaller than the length of the filelist ({len(filelist)}) and equal to or greater than 0.')
        if Nend < Nstart:
            raise Exception(f'Nstart ({Nstart}) should be smaller than Nend ({Nend}).')            
        if Nend > len(filelist):
            Nend = len(filelist)
        filelist=filelist[Nstart:Nend]
    construct_df2d(filelist,outpath) 