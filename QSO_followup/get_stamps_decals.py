#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
#import time
from glob import glob, iglob
from subprocess import call

import numpy as np
try:
  from urllib2 import urlopen #python2
except ImportError:
  from urllib.request import urlopen #python3
  from urllib.error import HTTPError
  from http.client import IncompleteRead


from astropy.io import ascii, fits
from astropy.table import Table

imgdic = {'first':'firstimage',
          'vla_stripe82':'stripe82image'}

EXAMPLES = """

Images can be viewed directly using the Sky viewer and raw data can be obtained through the NOAO portal (see also the information near the bottom of the files page).

Sections of the Legacy Survey can be obtained as JPEGs or FITS files using the cutout service as follows:

JPEG: http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2/?ra=244.6961&dec=7.4106&size=512&pixscale=0.27&bands=grz

FITS: http://legacysurvey.org/viewer/fits-cutout-decals-dr2?ra=002.79290&dec=19.28091&pixscale=0.911&size=512&bands=r

where "bands" is a string like "grz","gz","g", etc. As of the writing of this documentation the maximum size for cutouts (in number of pixels) is 512. Pixscale=0.262 will return (approximately) the native DECam pixels.

EXAMPLES

get_stamps_decals.py -i ydrops_test.csv  --bands r


           """


#function that build the url from the coordinate
def build_url(ra, dec, pixscale, npix, bands):
    # Updated to use https and match the viewer at https://www.legacysurvey.org/viewer/
    base = "https://www.legacysurvey.org/viewer/fits-cutout-decals-dr10?"
    ra = "ra={:0}&".format(ra)
    dec = "dec={:0}&".format(dec)
    pixscale = "pixscale={0:f}&".format(pixscale)
    size = "size={:0}&".format(npix)
    bands = "bands={0:s}".format(bands)
    
    url = base + ra + dec +pixscale + size + bands + "&FITS=1&Download=1"
    
    return url

	
	
def get_name(d):
    '''
    get name column - expects ls_id, with fallback to other common names
    '''
    try:
        n = d['ls_id']
    except KeyError:
        try:
            n = d['ps1_name']
        except KeyError:
            try:
                n = d['name']
            except KeyError:
                try:
                    n = d['NAME']
                except KeyError:
                    try:
                        n = d['ID']
                    except KeyError:
                        raise KeyError("Could not find name column. Expecting 'ls_id', 'ps1_name', 'name', 'NAME', or 'ID'")
        
    return n

def get_coordinates(d):
    '''
    get ra and dec columns handling different cases
    '''
    # Try various column name combinations
    ra_cols = ['ra', 'RA', 'RAJ2000', 'ra_deg', 'RA_deg']
    dec_cols = ['dec', 'DEC', 'DEJ2000', 'dec_deg', 'DEC_deg', 'de']
    
    ra = None
    dec = None
    
    # Try lowercase first
    for ra_col in ra_cols:
        if ra_col.lower() in [col.lower() for col in d.colnames]:
            # Find the actual column name (case-sensitive match)
            for col in d.colnames:
                if col.lower() == ra_col.lower():
                    ra = d[col]
                    break
            if ra is not None:
                break
    
    for dec_col in dec_cols:
        if dec_col.lower() in [col.lower() for col in d.colnames]:
            # Find the actual column name (case-sensitive match)
            for col in d.colnames:
                if col.lower() == dec_col.lower():
                    dec = d[col]
                    break
            if dec is not None:
                break
    
    if ra is None or dec is None:
        available_cols = ", ".join(d.colnames)
        raise KeyError("Could not find coordinate columns. Available columns: {0}. Expecting 'ra'/'dec' or 'RA'/'DEC'".format(available_cols))
    
    return ra, dec
    

#function that take the arguments in input
def parse_arguments():

    parser = argparse.ArgumentParser(
        description='''

	Get a list with coordinates and ps1 name and extract the first image,
	as a ps1_name_first.fits file.

        ''',
	formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('-i','--input', required=True, type=str, 
                            help='Input file (FITS or ASCII) containing\
                            the target coordinates and name\
		            in columns: ra, dec, ls_id' )

    parser.add_argument('--pixscale', required=False, type=float, 
                        default=0.262,
                            help='The pixels scale in arcsec per pixel. The default pixel scale 0.262 will \
                            return (approximately) the native DECam pixels.' )

    parser.add_argument('--npix', required=False, type=int, 
                        default=512,
                            help='The number of pixels on a side of the image. As of the writing of this documentation the maximum\
                             size for cutouts (in number of pixels) is 512. This combined with the native DECam scale 0.262 yields an image of approx 2x2 arcmin ' )
    
    parser.add_argument('--bands', required=False, type=str, 
                        default="z", choices=["g", "r", "z"], 
                            help='Filter' )
    

    return parser.parse_args()   

if __name__ == '__main__':
    args=parse_arguments()
    
    # Read input file - handle both FITS and ASCII files
    if args.input.endswith('.fits') or args.input.endswith('.fit'):
        # Read FITS file
        try:
            fits_data = fits.getdata(args.input, ext=0)
            data = Table(fits_data)
        except Exception as e1:
            try:
                # Try extension 1 if extension 0 fails
                print("No data in HDU 0, trying HDU 1...")
                fits_data = fits.getdata(args.input, ext=1)
                data = Table(fits_data)
            except Exception as e2:
                print("Error reading FITS file {0}".format(args.input))
                print("Extension 0 error: {0}".format(e1))
                print("Extension 1 error: {0}".format(e2))
                import sys
                sys.exit(1)
    else:
        # Read ASCII file using astropy
        # Handle files with comment header lines (starting with #)
        try:
            # Read the first line to check if it's a comment header
            with open(args.input, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('#'):
                # Extract column names from comment line
                header_line = first_line.lstrip('#').strip()
                col_names = header_line.split()
                # Remove trailing commas from column names
                col_names = [name.rstrip(',') for name in col_names]
                
                # Read data without header, then set column names
                data = ascii.read(args.input, format='basic', header_start=None, data_start=1, comment='#')
                
                # Rename columns if we have the right number
                if len(col_names) == len(data.colnames):
                    data.rename_columns(data.colnames, col_names)
                else:
                    print("Warning: Number of header columns ({0}) doesn't match data columns ({1})".format(
                        len(col_names), len(data.colnames)))
                    print("Header columns:", col_names)
                    print("Data columns:", data.colnames)
            else:
                # No comment header, try normal reading
                data = ascii.read(args.input)
        except Exception as e1:
            # If auto-detection fails, try common formats
            try:
                data = ascii.read(args.input, format='basic')
            except Exception as e2:
                try:
                    data = ascii.read(args.input, format='fixed_width')
                except Exception as e3:
                    print("Error reading input file {0}".format(args.input))
                    print("Primary error: {0}".format(e1))
                    print("Basic format error: {0}".format(e2))
                    print("Fixed-width error: {0}".format(e3))
                    import sys
                    sys.exit(1)
    
    # Handle different column name cases
    ra, dec = get_coordinates(data)
    ps1name = get_name(data)
    i = 0.
    j = 0

    for ra, dec_i, ps1name_i in zip(ra, dec, ps1name):
        print("*" * 30)
        print("Getting DECalS image for ", ps1name_i)
        print("bands: ", args.bands)
        i = i+1.
        url = build_url(ra, dec_i, pixscale=args.pixscale, npix=args.npix, bands=args.bands)
        print("=" * 30 )
        print("Obtaining image from")
        print(url)
	 
        
 
        try:
            datafile = urlopen(url)
            file = datafile.read()
            name_fits = '{0}_decals_{1}.fits'.format(str(ps1name_i), args.bands)
            output = open(name_fits,'wb')
            output.write(file)
            output.close()
            print(name_fits, " created")
        except (IncompleteRead, HTTPError) as err:
            print(err)
            msg = "No DECalS coverage for " + str(ps1name_i) + " in bands: " + args.bands
            print(msg)
            j+=1
        
      
           
    print("Objects without images in DECALS in some bands: ", j, " of ", i)
           
      
