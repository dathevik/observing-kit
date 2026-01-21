#!/usr/bin/env python
"""
Real WCS Finding Charts (Compatibility Workaround)
Uses actual astronomical coordinates from FITS files
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
import os

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
            try:
                ls_id = d['ps1_name']
            except KeyError:
                try:
                    ls_id = d['name']
                except KeyError:
                    try:
                        ls_id = d['NAME']
                    except KeyError:
                        try:
                            ls_id = d['ID']
                        except KeyError:
                            raise KeyError("Could not find ls_id column. Expecting 'ls_id', 'LS_ID', 'ps1_name', 'name', 'NAME', or 'ID'")
    
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

def read_input_file(filename):
    '''
    Read input .dat file and extract ra, dec, and ls_id columns
    Handles comment headers (lines starting with #)
    '''
    try:
        # Read the first line to check if it's a comment header
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('#'):
            # Extract column names from comment line
            header_line = first_line.lstrip('#').strip()
            col_names = header_line.split()
            # Remove trailing commas from column names
            col_names = [name.rstrip(',') for name in col_names]
            
            # Read data without header, starting from line 1 (first data line)
            # data_start=0 means start from the first line, but we skip the comment header
            # So we need data_start=1 to start from the first data line (0-indexed: line 1)
            data = ascii.read(filename, format='basic', header_start=None, data_start=0, comment='#')
            
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
            data = ascii.read(filename)
    except Exception as e1:
        # If auto-detection fails, try common formats
        try:
            data = ascii.read(filename, format='basic')
        except Exception as e2:
            try:
                data = ascii.read(filename, format='fixed_width')
            except Exception as e3:
                print("Error reading input file {0}".format(filename))
                print("Primary error: {0}".format(e1))
                print("Basic format error: {0}".format(e2))
                print("Fixed-width error: {0}".format(e3))
                import sys
                sys.exit(1)
    
    # Extract columns
    ra, dec = get_coordinates(data)
    ls_id = get_ls_id(data)
    
    # Convert to list of tuples: (ra, dec, ls_id)
    sources = [(float(ra_i), float(dec_i), str(ls_id_i)) 
               for ra_i, dec_i, ls_id_i in zip(ra, dec, ls_id)]
    
    return sources

def create_real_wcs_chart(ra, dec, source_id):
    """
    Create finding chart using REAL WCS from FITS but avoiding matplotlib projection bug
    """
    
    print(f"Creating real WCS chart for {source_id}...")
    
    # Find stamp file - check multiple locations
    ps1_locations = [
        f"{source_id}_ps1_z.fits",  # Current directory
        f"stamps_ps1/{source_id}_ps1_z.fits",
        f"Gemini_South/stamps_panstars/{source_id}_ps1_z.fits"
    ]
    decals_locations = [
        f"{source_id}_decals_z.fits",  # Current directory
        f"stamps_decals/{source_id}_decals_z.fits",
        f"Gemini_South/stamps_decals/{source_id}_decals_z.fits"
    ]
    
    stamp_file = None
    survey = None
    
    # Check PS1 files first
    for ps1_file in ps1_locations:
        if os.path.exists(ps1_file):
            stamp_file = ps1_file
            survey = "Pan-STARRS"
            break
    
    # If no PS1 file, check DECaLS files
    if stamp_file is None:
        for decals_file in decals_locations:
            if os.path.exists(decals_file):
                stamp_file = decals_file
                survey = "DECaLS"
                break
    
    if stamp_file is None:
        print(f"✗ No stamp found for {source_id}")
        return False
    
    try:
        # Read FITS file and get the REAL WCS
        with fits.open(stamp_file) as hdul:
            image_data = hdul[0].data
            wcs = WCS(hdul[0].header)  # This is the REAL astronomical coordinate system!
        
        image_data = np.nan_to_num(image_data, nan=0.0)
        
        # Compute accurate pixel scales from REAL WCS (deg/pix -> arcsec/pix)
        # This accounts for CD/PC matrices and rotation/skew
        scales_deg = proj_plane_pixel_scales(wcs)  # [dx_deg/pix, dy_deg/pix]
        pixel_scale_arcsec_x = abs(scales_deg[0]) * 3600.0
        pixel_scale_arcsec_y = abs(scales_deg[1]) * 3600.0

        # Zoom to ~1.5 arcmin FOV around the center (crop)
        fov_arcmin = 1.5
        arcmin_pixels_x = (fov_arcmin * 60.0) / pixel_scale_arcsec_x
        arcmin_pixels_y = (fov_arcmin * 60.0) / pixel_scale_arcsec_y
        cx_full, cy_full = image_data.shape[1] // 2, image_data.shape[0] // 2
        half_w = int(round(arcmin_pixels_x / 2.0))
        half_h = int(round(arcmin_pixels_y / 2.0))
        x0 = max(0, cx_full - half_w)
        x1 = min(image_data.shape[1], cx_full + half_w)
        y0 = max(0, cy_full - half_h)
        y1 = min(image_data.shape[0], cy_full + half_h)

        img = image_data[y0:y1, x0:x1]

        # Create regular matplotlib plot (avoid projection=wcs to prevent bug)
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display cropped image (inverted grayscale to match reference)
        vmin, vmax = np.percentile(img[img != 0], [2, 98]) if np.any(img != 0) else (img.min(), img.max())
        ax.imshow(img, cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower')

        # Now use the REAL WCS to create proper coordinate ticks for the cropped view
        ny, nx = img.shape
        
        # Create coordinate arrays using REAL WCS
        # Important: use absolute pixel indices so WCS returns the REAL coordinates
        x_pixels = np.arange(x0, x1)
        y_pixels = np.arange(y0, y1)
        
        # Convert pixel coordinates to REAL world coordinates using WCS
        xx, yy = np.meshgrid(x_pixels, y_pixels)
        world_coords = wcs.pixel_to_world(xx, yy)
        
        # Extract REAL RA and Dec arrays
        ra_array = world_coords.ra.deg
        dec_array = world_coords.dec.deg
        
        # Create proper tick positions using REAL coordinates
        # Use symmetric offsets around the center in arcsec to mimic the example
        # RA increases to the left; DEC increases upward
        # Generate ticks within the cropped FOV every 15 arcsec
        half_width_arcsec_x = (fov_arcmin * 60.0) / 2.0
        half_width_arcsec_y = (fov_arcmin * 60.0) / 2.0
        step_arcsec = 15
        ra_offsets_arcsec = [o for o in range(-int(half_width_arcsec_x), int(half_width_arcsec_x) + 1, step_arcsec)]
        dec_offsets_arcsec = [o for o in range(-int(half_width_arcsec_y), int(half_width_arcsec_y) + 1, step_arcsec)]

        # Pixel scale from WCS (deg/pix -> arcsec/pix); use absolute values for spacing
        ra_pix_per_arcsec = 1.0 / (abs(wcs.wcs.cdelt[0]) * 3600.0)
        dec_pix_per_arcsec = 1.0 / (abs(wcs.wcs.cdelt[1]) * 3600.0)

        # Center pixel
        cx, cy = nx // 2, ny // 2

        # Compute RA tick pixel positions and labels in sexagesimal
        ra_tick_positions = []
        ra_tick_labels = []
        for off in ra_offsets_arcsec:
            # RA increases to the left on images; subtract the offset in pixels
            px = cx - off * ra_pix_per_arcsec
            if 0 <= px <= nx - 1:
                ra_tick_positions.append(px)
                # Get REAL RA at bottom row for this pixel
                real_ra_deg = ra_array[0, int(round(px))]
                if off == 0:
                    # Full h:m:s for the center
                    ra_hms = SkyCoord(real_ra_deg * u.deg, dec * u.deg).ra.to_string(
                        unit=u.hour, sep=': ', precision=0, pad=True
                    ).replace(': ', 'h', 1).replace(': ', 'm') + 's'
                    ra_tick_labels.append(ra_hms)
                else:
                    # Seconds offset label like "+30s" / "-30s"
                    sign = '+' if off > 0 else ''
                    ra_tick_labels.append(f"{sign}{abs(off):.0f}s")

        # Compute Dec tick pixel positions and labels in sexagesimal
        dec_tick_positions = []
        dec_tick_labels = []
        for off in dec_offsets_arcsec:
            py = cy + off * dec_pix_per_arcsec  # DEC increases upward
            if 0 <= py <= ny - 1:
                dec_tick_positions.append(py)
                real_dec_deg = dec_array[int(round(py)), 0]
                if off == 0:
                    dec_dms = SkyCoord(ra * u.deg, real_dec_deg * u.deg).dec.to_string(
                        unit=u.degree, sep=': ', precision=0, pad=True, alwayssign=True
                    ).replace(': ', '°', 1).replace(': ', "'") + '"'
                    dec_tick_labels.append(dec_dms)
                else:
                    sign = '+' if off > 0 else ''
                    # Show arcmin/arcsec as in example; off is in arcsec
                    if abs(off) >= 60:
                        dec_tick_labels.append(f"{sign}{abs(off)/60:.0f}'")
                    else:
                        dec_tick_labels.append(f"{sign}{abs(off):.0f}\"")

        # Apply ticks and labels
        ax.set_xticks(ra_tick_positions)
        ax.set_xticklabels(ra_tick_labels, fontsize=12)
        ax.set_yticks(dec_tick_positions)
        ax.set_yticklabels(dec_tick_labels, fontsize=12)
        
        # Add coordinate grid
        ax.grid(True, color='black', ls='-', lw=1, alpha=0.8)
        
        # Labels
        ax.set_xlabel('Right Ascension', fontsize=14, weight='bold')
        ax.set_ylabel('Declination', fontsize=14, weight='bold')
        
        # Mark target using REAL WCS coordinate transformation
        target_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        target_pix = wcs.world_to_pixel(target_coord)  # REAL transformation!
        
        # Red circle at target (convert absolute pixels to cropped frame)
        circle = plt.Circle((target_pix[0] - x0, target_pix[1] - y0), radius=20,
                           fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(circle)
        
        # Title showing REAL coordinates
        title = f"RA = {ra:.6f}° ; DEC = {dec:+.6f}° ({survey})"
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        
        # Scale bar using REAL pixel scale from WCS
        scales_deg = proj_plane_pixel_scales(wcs)
        pixel_scale_deg_x = abs(scales_deg[0])  # deg/pix along x
        arcmin_pixels = (1.0/60.0) / pixel_scale_deg_x  # pixels per arcminute (x-axis)
        
        # Draw scale bar
        x_center, y_bottom = nx//2, ny*0.1
        ax.plot([x_center - arcmin_pixels/2, x_center + arcmin_pixels/2], 
               [y_bottom, y_bottom], 'k-', linewidth=5)
        
        # Scale label
        ax.text(x_center, y_bottom - ny*0.04, '1 arcmin', 
               ha='center', va='top', fontsize=16, weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        
        # Save
        filename = f"finding_chart_real_{source_id}.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate finding charts using REAL WCS from FITS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  python finding_charts_real.py -i candidates_Jan.dat

The input file should contain columns: ra, dec, and ls_id (or variations).
        ''')
    
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .dat file containing ra, dec, and ls_id columns')
    
    args = parser.parse_args()
    
    print("Real WCS Finding Chart Generator (Compatibility Fixed)")
    print("Using actual astronomical coordinates from FITS WCS")
    print("=" * 60)
    
    # Read sources from input file
    print(f"Reading sources from: {args.input}")
    try:
        sources = read_input_file(args.input)
        print(f"Found {len(sources)} sources")
    except Exception as e:
        print(f"Error reading input file: {e}")
        import sys
        sys.exit(1)
    
    if len(sources) == 0:
        print("No sources found in input file!")
        import sys
        sys.exit(1)
    
    successful = failed = 0
    
    for i, (ra, dec, source_id) in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] {source_id}...")
        
        if create_real_wcs_chart(ra, dec, source_id):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Completed! {successful} charts created using REAL WCS coordinates.")
    if failed > 0:
        print(f"{failed} failed.")
