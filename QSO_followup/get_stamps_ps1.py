#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import requests
import os
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
#import time

from glob import glob, iglob
from subprocess import call

from astropy.io import ascii, fits
from astropy.table import Table
from lsdlib import read_input

from astropy.visualization import PercentileInterval, AsinhStretch

from PIL import Image
from io import BytesIO


EXAMPLES = """

EXAMPLES

get_stamps_ps1 -i ../decals_may2016_test1_good_wiseYJ.fits --band z

           """
	
def get_ls_id(d):
    '''
    get ls_id column handling different cases
    '''
    try:
        ls_id = d['ls_id']
    except KeyError:
        try:
            ls_id = d['LS_ID']
        except KeyError:
            raise KeyError("Could not find ls_id column. Expecting 'ls_id' or 'LS_ID'")
        
    return ls_id

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
    
# From example: https://ps1images.stsci.edu/ps1image.html
def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

# From example: https://ps1images.stsci.edu/ps1image.html
def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel) 240=60arcsec=1arcmin
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    
    # Check if table is empty (no images found)
    if len(table) == 0:
        print(f"Warning: No PS1 images found for RA={ra}, DEC={dec}, filters={filters}")
        return []
    
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    print(url)
    return url

#function that take the arguments in input
def parse_arguments():

    parser = argparse.ArgumentParser(
        description='''

	Get a list with coordinates and ps1 name and extract the PS1 image,
	as a ps1_name_ps1_band.fits file.

        ''',
	formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument('-i','--input', required=True, type=str, 
                            help='File containing\
                            the target coordinates and ls_id\
		            in columns called\
                            ra, dec, and ls_id' )

    parser.add_argument('--image_width', required=False, type=float, 
                        default=1.,
                            help='The image width in arcmin. (default: %(default)s)' )

    parser.add_argument('--band', required=False, type=str, 
                        default="z", choices=['g', 'r', 'i', 'z', 'y'], 
                            help='Filter (default: %(default)s)' )

    parser.add_argument('-e','--ext', required=False, default=None, type=int,
                            help='If a fits file, the extension to read. None otherwise' )
    

    return parser.parse_args()   


if __name__ == '__main__':
    args=parse_arguments()
    
    if args.ext is not None:
        data = read_input(args.input, ext=args.ext)
    else:
        # Try extension 0 first, if it fails try extension 1
        try:
            data = read_input(args.input, ext=0)
        except (IndexError, KeyError):
            try:
                print("No data in HDU 0, trying HDU 1...")
                data = read_input(args.input, ext=1)
            except Exception as e:
                print("Error reading input file: {0}".format(e))
                import sys
                sys.exit(1)
    
    # Handle different column name cases
    ra, dec = get_coordinates(data)
    ls_id = get_ls_id(data)
    i = 0.
    print("bands: ", args.band)
    successful = 0
    failed = 0
    
    for ra_i, dec_i, ls_id_i in zip(ra, dec, ls_id):
        print("*" * 30)
        print("Getting PS1 image for ls_id: ", ls_id_i)
        print("RA: {0:.6f} deg, DEC: {1:.6f} deg".format(ra_i, dec_i))
             
        url_i = geturl(ra_i,dec_i,size=480,output_size=None,filters=args.band,format='fits',color=False)
        
        # Check if URL list is empty (no images found)
        if not url_i or len(url_i) == 0:
            print("No PS1 images available for this object in band {0}".format(args.band))
            failed += 1
            continue
        
        outname_i = str(ls_id_i) + '_ps1_' + args.band + '.fits'
        try:
            fh = fits.open(url_i[0])
            fh.writeto(outname_i, overwrite=True)
            print(outname_i,"created")
            successful += 1
        except Exception as e:
            print("Error downloading or saving {0}: {1}".format(outname_i, e))
            failed += 1
        #fim = fh[0].data        
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Total processed: {0}".format(len(ra)))
    print("Successful: {0}".format(successful))
    print("Failed: {0}".format(failed))        
           
      
