#!/usr/bin/env python
from __future__ import print_function, division
import json
import argparse
import os
import time
try:
    from urllib2 import urlopen  # python2
    from urllib import quote
except ImportError:
    from urllib.request import urlopen  # python3
    from urllib.parse import quote

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

def fetch_racs_json(url):
    """
    Fetch JSON output from CASDA cutout API URL.
    
    Parameters:
    -----------
    url : str
        The CASDA API URL to fetch JSON from
    
    Returns:
    --------
    dict : JSON data as a dictionary
    """
    try:
        response = urlopen(url)
        data = response.read()
        json_data = json.loads(data.decode('utf-8'))
        return json_data
    except Exception as e:
        print("Error fetching JSON: {0}".format(e))
        return None

def format_coord_for_url(ra_deg, dec_deg):
    """
    Convert RA and DEC from degrees to sexagesimal format for CASDA URL.
    
    Parameters:
    -----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    
    Returns:
    --------
    tuple : (ra_str, dec_str) formatted as HH:MM:SS.sss and DD:MM:SS.sss
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    
    # Format RA as HH:MM:SS.sss
    ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
    
    # Format DEC as DD:MM:SS.sss (always include sign)
    dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=3, pad=True, alwayssign=True)
    
    return ra_str, dec_str

def build_url(ra_deg, dec_deg, size=0.0667, surveys=None):
    """
    Build CASDA cutout API URL from coordinates.
    
    Parameters:
    -----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    size : float
        Cutout size in degrees (default: 0.0667)
    surveys : list
        List of survey codes (default: ['RACS-Low', 'RACS-Mid', 'RACS-High'])
    
    Returns:
    --------
    str : Complete CASDA API URL
    """
    if surveys is None:
        surveys = ['RACS-Low', 'RACS-Mid', 'RACS-High']
    
    ra_str, dec_str = format_coord_for_url(ra_deg, dec_deg)
    
    base_url = "https://data.csiro.au/dap/ws/v2/domains/casdaCutout/cutout?"
    
    # Add survey parameters
    survey_params = "&".join(["surveys={0}".format(survey) for survey in surveys])
    
    # URL encode the coordinates
    ra_encoded = quote(ra_str)
    dec_encoded = quote(dec_str)
    
    url = "{0}{1}&dec={2}&ra={3}&size={4}".format(
        base_url, survey_params, dec_encoded, ra_encoded, size
    )
    
    return url

def get_coordinates(d):
    '''
    get ra and dec columns handling different cases
    '''
    try:
        ra = d['ra']
        dec = d['dec']
    except KeyError:
        try:
            ra = d['RA']
            dec = d['DEC']
        except KeyError:
            raise KeyError("Could not find coordinate columns. Expecting 'ra'/'dec' or 'RA'/'DEC'")
    
    return ra, dec

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

def download_fits(url, output_path):
    """
    Download a FITS file from a URL and save it to the specified path.
    
    Parameters:
    -----------
    url : str
        URL of the FITS file to download
    output_path : str
        Path where the FITS file should be saved
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        response = urlopen(url)
        data = response.read()
        with open(output_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print("    Error downloading {0}: {1}".format(url, e))
        return False

def wait_cooldown(seconds=10):
    """
    Wait for a cooldown period between requests.
    
    Parameters:
    -----------
    seconds : float
        Number of seconds to wait (default: 10)
    """
    print("    Waiting {0} seconds before next request...".format(seconds))
    time.sleep(seconds)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Get RACS cutout JSON data for objects in observations_gemini.dat')
    
    parser.add_argument('-i', '--input', required=False, type=str,
                        default='observations_gemini.dat',
                        help='Input file (default: observations_gemini.dat)')
    
    parser.add_argument('--size', required=False, type=float,
                        default=0.0667,
                        help='Cutout size in degrees (default: 0.0667)')
    
    parser.add_argument('--test', action='store_true',
                        help='Test mode: fetch JSON for only the first row')
    
    parser.add_argument('--output-dir', required=False, type=str,
                        default='stamps_racs',
                        help='Output directory for FITS files (default: stamps_racs)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
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
        try:
            data = ascii.read(args.input)
        except Exception as e1:
            try:
                data = ascii.read(args.input, format='basic')
            except Exception as e2:
                try:
                    data = ascii.read(args.input, format='fixed_width')
                except Exception as e3:
                    print("Error reading input file {0}".format(args.input))
                    print("Auto-detection error: {0}".format(e1))
                    print("Space-separated error: {0}".format(e2))
                    print("Fixed-width error: {0}".format(e3))
                    import sys
                    sys.exit(1)
    
    # Handle different column name cases
    ra, dec = get_coordinates(data)
    ls_id = get_ls_id(data)
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: {0}".format(output_dir))
    
    # Determine how many rows to process
    if args.test:
        num_rows = 1
        print("=" * 60)
        print("TEST MODE: Processing only the first row")
        print("=" * 60)
    else:
        num_rows = len(data)
        print("=" * 60)
        print("Processing {0} rows from {1}".format(num_rows, args.input))
        print("=" * 60)
    
    successful = 0
    failed = 0
    total_downloaded = 0
    total_failed_downloads = 0
    
    for i in range(num_rows):
        ra_i = ra[i]
        dec_i = dec[i]
        ls_id_i = ls_id[i]
        
        print("\n" + "*" * 60)
        print("Row {0}/{1}: RA={2:.6f} deg, DEC={3:.6f} deg, ls_id={4}".format(
            i+1, num_rows, ra_i, dec_i, ls_id_i))
        
        url = build_url(ra_i, dec_i, size=args.size)
        print("URL: {0}".format(url))
        print("-" * 60)
        
        json_data = fetch_racs_json(url)
        
        if json_data:
            print("Successfully fetched JSON data")
            print("Number of surveys: {0}".format(len(json_data)))
            
            # Create subdirectory for this ls_id
            ls_id_dir = os.path.join(output_dir, str(ls_id_i))
            if not os.path.exists(ls_id_dir):
                os.makedirs(ls_id_dir)
                print("Created directory: {0}".format(ls_id_dir))
            
            # Download FITS files from each survey's cutouts
            for survey_idx, survey in enumerate(json_data):
                survey_code = survey['surveyCode']
                cutouts = survey.get('cutouts', [])
                print("  - {0}: {1} cutout(s)".format(survey_code, len(cutouts)))
                
                # Download each cutout
                for idx, cutout in enumerate(cutouts):
                    data_url = cutout.get('dataUrl')
                    if not data_url:
                        print("    Warning: No dataUrl found for cutout {0}".format(idx))
                        continue
                    
                    # Build filename: {ls_id}_{survey}_{index}.fits
                    # Only add index if there are multiple cutouts
                    if len(cutouts) > 1:
                        filename = "{0}_{1}_{2}.fits".format(ls_id_i, survey_code, idx)
                    else:
                        filename = "{0}_{1}.fits".format(ls_id_i, survey_code)
                    
                    output_path = os.path.join(ls_id_dir, filename)
                    
                    # Check if file already exists
                    if os.path.exists(output_path):
                        print("    Skipping {0} (already exists)".format(filename))
                        total_downloaded += 1
                        # Still wait even if skipping (to respect rate limits)
                        is_last = (i == num_rows - 1 and 
                                  survey_idx == len(json_data) - 1 and 
                                  idx == len(cutouts) - 1)
                        if not is_last:
                            wait_cooldown(10)
                        continue
                    
                    print("    Downloading {0}...".format(filename))
                    if download_fits(data_url, output_path):
                        print("    Saved: {0}".format(output_path))
                        total_downloaded += 1
                    else:
                        total_failed_downloads += 1
                    
                    # Wait 10 seconds after each FITS download (except for the very last one)
                    is_last = (i == num_rows - 1 and 
                              survey_idx == len(json_data) - 1 and 
                              idx == len(cutouts) - 1)
                    if not is_last:
                        wait_cooldown(10)
            
            successful += 1
        else:
            print("Failed to fetch JSON data")
            failed += 1
        
        # Wait 10 seconds after JSON fetch before moving to next row (except after last row)
        if i < num_rows - 1:
            wait_cooldown(10)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Total processed: {0}".format(num_rows))
    print("Successful JSON fetches: {0}".format(successful))
    print("Failed JSON fetches: {0}".format(failed))
    print("Total FITS files downloaded: {0}".format(total_downloaded))
    print("Total FITS download failures: {0}".format(total_failed_downloads))
    print("Output directory: {0}".format(output_dir))

