import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy.table import Table
import os
import glob
import argparse
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
try:
    from reproject import reproject_interp
except ImportError:
    raise ImportError("The 'reproject' module is required. Please install it with: pip install reproject")
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from astropy.wcs.utils import proj_plane_pixel_scales


def create_radio_optical_overlay(radio_filenames, optical_filenames, coords_source, fig_savename, image_size_asec = 60., optical_label = None, ls_id = None):

	# Create a larger figure - one subplot per source, arranged vertically
	num_sources = len(radio_filenames)
	f = plt.figure(figsize=(24, 8*num_sources))
	
	# Add figure title with ls_id if provided
	if ls_id is not None:
		f.suptitle('ls_id: {0}'.format(ls_id), fontsize=18, fontweight='bold', y=0.995)
	
	# Create subplots - one row per source, showing optical, radio, and overlay (3 columns)
	gs1 = gridspec.GridSpec(num_sources, 3, figure=f, width_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
	    
	# Loop over all image files
	for i, radio_file in enumerate(radio_filenames):

		# Open radio data
		radio_cut = fits.open(radio_file)
		radio_data = np.squeeze(radio_cut[0].data)  # Remove any extra dimensions
		wcs_radio_cut = WCS(radio_cut[0].header, naxis=2)

		# Open optical data
		opt_cut = fits.open(optical_filenames[i])
		optical_data = np.squeeze(opt_cut[0].data)  # Remove any extra dimensions

		lae1_optical = Cutout2D(data=optical_data,
		              position=coords_source[i],
		              size= image_size_asec*u.arcsec,
		              wcs=WCS(opt_cut[0].header))

		# Reproject the radio data
		lae1_radio = Cutout2D(data=radio_data, position=coords_source[i], size=image_size_asec*u.arcsec, wcs=wcs_radio_cut) 
		radio_reproj = reproject_interp(input_data=(lae1_radio.data, lae1_radio.wcs), output_projection=lae1_optical.wcs, shape_out=lae1_optical.shape)
		lae1_radio_reproj = np.ma.array(radio_reproj[0], mask=1-radio_reproj[1])

		radio_rms = mad_std(lae1_radio.data)
		rms_opt = mad_std(lae1_optical.data)

		# Extract beam information from radio header
		beam_maj = radio_cut[0].header.get('BMAJ', None)  # Major axis in degrees
		beam_min = radio_cut[0].header.get('BMIN', None)  # Minor axis in degrees
		beam_pa = radio_cut[0].header.get('BPA', 0.0)  # Position angle in degrees
		
		# Calculate pixel scale for scale bar
		pixel_scales_opt = proj_plane_pixel_scales(lae1_optical.wcs)  # Returns [dx, dy] in degrees
		pixel_scale_arcsec = abs(pixel_scales_opt[0]) * 3600.0  # Convert to arcsec/pixel
		
		# Helper function to add beam ellipse
		def add_beam(ax, wcs, beam_maj, beam_min, beam_pa, pixel_scale_arcsec, position='bottom left'):
			if beam_maj is None or beam_min is None:
				return
			
			# Convert beam from degrees to arcseconds, then to pixels
			beam_maj_arcsec = beam_maj * 3600.0  # Convert deg to arcsec
			beam_min_arcsec = beam_min * 3600.0
			beam_maj_pix = beam_maj_arcsec / pixel_scale_arcsec
			beam_min_pix = beam_min_arcsec / pixel_scale_arcsec
			
			# Position in lower left corner (in pixel coordinates)
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			x_pos = xlim[0] + 0.08 * (xlim[1] - xlim[0])
			y_pos = ylim[0] + 0.08 * (ylim[1] - ylim[0])
			
			# Create ellipse for beam
			beam_ellipse = Ellipse((x_pos, y_pos), beam_maj_pix, beam_min_pix, 
			                       angle=beam_pa, fill=True, facecolor='white', 
			                       edgecolor='black', linewidth=1.5, zorder=10)
			ax.add_patch(beam_ellipse)
			
			# Add label
			ax.text(x_pos, y_pos - beam_maj_pix*0.8, 'Beam', 
			       color='white', fontsize=10, ha='center', va='top',
			       weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
			                               facecolor='black', alpha=0.7, edgecolor='white'))
		
		# Helper function to add scale bar
		def add_scale_bar(ax, wcs, pixel_scale_arcsec, position='bottom right'):
			# Choose scale bar size (10 arcsec)
			scale_arcsec = 10.0
			scale_pixels = scale_arcsec / pixel_scale_arcsec
			
			# Position in lower right corner
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			x_end = xlim[1] - 0.05 * (xlim[1] - xlim[0])
			x_start = x_end - scale_pixels
			y_pos = ylim[0] + 0.05 * (ylim[1] - ylim[0])
			
			# Draw scale bar
			ax.plot([x_start, x_end], [y_pos, y_pos], 
			       'w-', linewidth=3, solid_capstyle='butt', zorder=10)
			
			# Add vertical ticks
			tick_height = 0.02 * (ylim[1] - ylim[0])
			ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], 
			       'w-', linewidth=3, zorder=10)
			ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], 
			       'w-', linewidth=3, zorder=10)
			
			# Label
			ax.text((x_start + x_end)/2, y_pos + tick_height*2, 
			       f"{scale_arcsec:.0f}\"", color='white', fontsize=11, 
			       ha='center', va='bottom', weight='bold',
			       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
			               alpha=0.7, edgecolor='white'))

		# Left panel: Optical image only
		Ax_opt = f.add_subplot(gs1[i, 0], projection=lae1_optical.wcs)
		Ax_opt.imshow(lae1_optical.data, vmin=-1*rms_opt, vmax=5*rms_opt, cmap='Greys', origin='lower')
		# Use provided label or default to DECaLS
		if optical_label is None:
			optical_label = 'DECaLS'
		Ax_opt.set_title('Optical ({0})'.format(optical_label), fontsize=14, pad=10)
		
		lon_opt = Ax_opt.coords[0]
		lat_opt = Ax_opt.coords[1]
		lon_opt.set_ticks(number=3)
		lat_opt.set_ticks(number=3)
		lon_opt.set_ticklabel(exclude_overlapping=True)
		lat_opt.set_ticklabel(exclude_overlapping=True)
		lat_opt.set_axislabel('Dec (J2000)', fontsize=12)
		lon_opt.set_axislabel('RA (J2000)', fontsize=12)
		
		# Add scale bar to optical panel
		add_scale_bar(Ax_opt, lae1_optical.wcs, pixel_scale_arcsec)

		# Middle panel: Radio image only
		Ax_radio = f.add_subplot(gs1[i, 1], projection=lae1_radio.wcs)
		Ax_radio.imshow(lae1_radio.data, vmin=-1*radio_rms, vmax=10*radio_rms, cmap='viridis', origin='lower')
		Ax_radio.set_title('Radio (RACS)', fontsize=14, pad=10)
		
		lon_radio = Ax_radio.coords[0]
		lat_radio = Ax_radio.coords[1]
		lon_radio.set_ticks(number=3)
		lat_radio.set_ticks(number=3)
		lon_radio.set_ticklabel(exclude_overlapping=True)
		lat_radio.set_ticklabel(exclude_overlapping=True)
		lat_radio.set_ticks_visible(False)
		lat_radio.set_ticklabel_visible(False)
		lon_radio.set_axislabel('RA (J2000)', fontsize=12)
		
		# Calculate pixel scale for radio WCS
		pixel_scales_radio = proj_plane_pixel_scales(lae1_radio.wcs)
		pixel_scale_arcsec_radio = abs(pixel_scales_radio[0]) * 3600.0
		
		# Add beam and scale bar to radio panel
		if beam_maj is not None and beam_min is not None:
			add_beam(Ax_radio, lae1_radio.wcs, beam_maj, beam_min, beam_pa, pixel_scale_arcsec_radio)
		add_scale_bar(Ax_radio, lae1_radio.wcs, pixel_scale_arcsec_radio)

		# Right panel: Radio contours over optical overlay
		Ax_overlay = f.add_subplot(gs1[i, 2], projection=lae1_optical.wcs)
		Ax_overlay.imshow(lae1_optical.data, vmin=-1*rms_opt, vmax=5*rms_opt, cmap='Greys', origin='lower', alpha=0.7)
		Ax_overlay.contour(lae1_radio_reproj.data, levels = np.array([1,2,3,5,10,20,50,100,200])*radio_rms, colors='maroon', linewidths=1.5, norm=LogNorm())
		Ax_overlay.set_title('Overlay (RACS + Optical)', fontsize=14, pad=10)
		
		lon_overlay = Ax_overlay.coords[0]
		lat_overlay = Ax_overlay.coords[1]
		lon_overlay.set_ticks(number=3)
		lat_overlay.set_ticks(number=3)
		lon_overlay.set_ticklabel(exclude_overlapping=True)
		lat_overlay.set_ticklabel(exclude_overlapping=True)
		lat_overlay.set_ticks_visible(False)
		lat_overlay.set_ticklabel_visible(False)
		lon_overlay.set_axislabel('RA (J2000)', fontsize=12)
		
		# Add beam and scale bar to overlay panel
		if beam_maj is not None and beam_min is not None:
			add_beam(Ax_overlay, lae1_optical.wcs, beam_maj, beam_min, beam_pa, pixel_scale_arcsec)
		add_scale_bar(Ax_overlay, lae1_optical.wcs, pixel_scale_arcsec)

	plt.savefig(fig_savename, dpi=300, format='png', bbox_inches='tight')
	print("Saved overlay image: {0}".format(fig_savename))
	plt.close()  # Close the figure to free memory


def get_optical_label_from_filename(filename):
	'''
	Determine the optical survey label from the filename.
	- If filename contains '_ps1', return 'PanSTARRS'
	- If filename contains 'decals', return 'DECALS'
	- Otherwise, return 'DECaLS' (default)
	'''
	filename_lower = filename.lower()
	if '_ps1' in filename_lower:
		return 'PanSTARRS'
	elif 'decals' in filename_lower:
		return 'DECALS'
	else:
		return 'DECaLS'


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

def find_files_by_ls_id(folder, ls_id, pattern='*'):
    """
    Find files in folder matching ls_id pattern.
    Searches in the folder and also in subdirectories named with ls_id.
    
    Parameters:
    -----------
    folder : str
        Directory to search in
    ls_id : str or int
        ls_id to search for
    pattern : str
        File pattern to match (e.g., '*.fits', '*_decals_*.fits')
    
    Returns:
    --------
    list : List of matching file paths
    """
    ls_id_str = str(ls_id)
    files = []
    
    # Check if folder path already ends with ls_id (e.g., stamps_racs/10995145654666845)
    folder_basename = os.path.basename(os.path.normpath(folder))
    is_ls_id_folder = (folder_basename == ls_id_str)
    
    if is_ls_id_folder:
        # If we're already in the ls_id folder, search for any .fits files
        if pattern == '*.fits' or pattern == '*':
            search_pattern = os.path.join(folder, '*.fits')
            files.extend(glob.glob(search_pattern))
        else:
            # Search with the pattern
            search_pattern = os.path.join(folder, ls_id_str + pattern)
            files.extend(glob.glob(search_pattern))
    else:
        # Search directly in the folder for files starting with ls_id
        search_pattern = os.path.join(folder, ls_id_str + pattern)
        files.extend(glob.glob(search_pattern))
        
        # Also search in subdirectory named with ls_id (if it exists)
        ls_id_subdir = os.path.join(folder, ls_id_str)
        if os.path.isdir(ls_id_subdir):
            # Search for any .fits files in that subdirectory
            if pattern == '*.fits' or pattern == '*':
                search_pattern_any = os.path.join(ls_id_subdir, '*.fits')
                files.extend(glob.glob(search_pattern_any))
            else:
                search_pattern_subdir = os.path.join(ls_id_subdir, ls_id_str + pattern)
                files.extend(glob.glob(search_pattern_subdir))
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Create radio-optical overlay images for specified sources')
    
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input table file (e.g., observations_gemini.dat)')
    
    parser.add_argument('--ls-id', required=False, type=str, nargs='+',
                        help='ls_id(s) of source(s) to process (can specify multiple). If not provided, processes all sources in the input file.')
    
    parser.add_argument('--racs-base-folder', required=False, type=str,
                        default='Gemini_South/stamps_racs',
                        help='Base folder containing RACS subdirectories (default: Gemini_South/stamps_racs)')
    
    parser.add_argument('--optical-folder', required=False, type=str,
                        default='Gemini_South/stamps_decals',
                        help='Folder containing optical FITS files (default: Gemini_South/stamps_decals)')
    
    parser.add_argument('--image-size', required=False, type=float,
                        default=60.0,
                        help='Image size in arcseconds (default: 60.0)')
    
    parser.add_argument('--radio-pattern', required=False, type=str,
                        default='*.fits',
                        help='File pattern for radio files (default: *.fits)')
    
    parser.add_argument('--optical-pattern', required=False, type=str,
                        default='*.fits',
                        help='File pattern for optical files (default: *.fits)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Read input table - handle both FITS and ASCII files
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
        # Read ASCII file
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
    
    # Get columns
    ra, dec = get_coordinates(data)
    ls_id_col = get_ls_id(data)
    
    # Convert ls_id column to strings for matching
    ls_id_col_str = [str(x) for x in ls_id_col]
    
    # Determine which ls_ids to process
    if args.ls_id is None:
        # Process all unique ls_ids in the input file
        ls_ids_to_process = sorted(list(set(ls_id_col_str)))
        print("=" * 60)
        print("Processing all sources in input file")
        print("Found {0} unique ls_id(s) to process".format(len(ls_ids_to_process)))
        print("=" * 60)
    else:
        # Process only specified ls_ids
        ls_ids_to_process = [str(x) for x in args.ls_id]
        print("=" * 60)
        print("Processing {0} specified ls_id(s)".format(len(ls_ids_to_process)))
        print("=" * 60)
    
    # Track processing statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Find matching rows for specified ls_ids
    # Process each ls_id separately to create individual overlay files
    for ls_id_query in ls_ids_to_process:
        radio_filenames = []
        optical_filenames = []
        coords_source = []
        
        ls_id_query_str = str(ls_id_query)
        
        # Find row index in table
        try:
            idx = ls_id_col_str.index(ls_id_query_str)
        except ValueError:
            print("Warning: ls_id {0} not found in table {1}".format(ls_id_query, args.input))
            skipped += 1
            continue
        
        # Get coordinates
        ra_i = ra[idx]
        dec_i = dec[idx]
        coord = SkyCoord(ra=ra_i * u.deg, dec=dec_i * u.deg, frame='icrs')
        coords_source.append(coord)
        
        # Automatically find radio folder with ls_id name
        radio_folder = os.path.join(args.racs_base_folder, ls_id_query_str)
        
        # If folder not found, try common alternative locations
        if not os.path.isdir(radio_folder):
            # Try Gemini_South/stamps_racs if not already there
            if 'Gemini_South' not in args.racs_base_folder:
                alt_folder = os.path.join('Gemini_South', args.racs_base_folder, ls_id_query_str)
                if os.path.isdir(alt_folder):
                    radio_folder = alt_folder
                    print("Found radio folder in alternative location: {0}".format(radio_folder))
                else:
                    print("Warning: Radio folder not found: {0}".format(radio_folder))
                    print("  Also tried: {0}".format(alt_folder))
                    print("  Searched in base folder: {0}".format(args.racs_base_folder))
                    skipped += 1
                    continue
            else:
                print("Warning: Radio folder not found: {0}".format(radio_folder))
                print("  Searched in base folder: {0}".format(args.racs_base_folder))
                skipped += 1
                continue
        
        print("Using radio folder: {0}".format(radio_folder))
        
        # Find all radio files
        radio_files = find_files_by_ls_id(radio_folder, ls_id_query, pattern=args.radio_pattern)
        if len(radio_files) == 0:
            print("Warning: No radio file found for ls_id {0} in {1}".format(ls_id_query, radio_folder))
            print("  Searched for pattern: {0}{1}".format(ls_id_query, args.radio_pattern))
            skipped += 1
            continue
        
        print("Found {0} radio file(s) for ls_id {1}".format(len(radio_files), ls_id_query))
        
        # Find optical file - handle both folder and direct file path
        if os.path.isfile(args.optical_folder):
            # If optical_folder is actually a file, use it directly
            print("Using optical file directly: {0}".format(args.optical_folder))
            optical_file_path = args.optical_folder
        else:
            # Otherwise, search in the folder
            optical_files = find_files_by_ls_id(args.optical_folder, ls_id_query, pattern=args.optical_pattern)
            if len(optical_files) == 0:
                print("Warning: No optical file found for ls_id {0} in {1}".format(ls_id_query, args.optical_folder))
                print("  Searched for pattern: {0}{1}".format(ls_id_query, args.optical_pattern))
                skipped += 1
                continue
            elif len(optical_files) > 1:
                print("Warning: Multiple optical files found for ls_id {0}, using first: {1}".format(ls_id_query, optical_files[0]))
            optical_file_path = optical_files[0]
        
        print("  Optical: {0}".format(optical_file_path))
        print("  Coordinates: RA={0:.6f} deg, DEC={1:.6f} deg".format(ra_i, dec_i))
        
        # Create output directory
        output_dir = "{0}_overlay".format(ls_id_query_str)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created output directory: {0}".format(output_dir))
        
        # Process each radio file
        for radio_file in radio_files:
            # Extract survey type and index from filename
            # Format: {ls_id}_{survey}_{index}.fits or {ls_id}_{survey}.fits
            radio_basename = os.path.basename(radio_file)
            # Remove .fits extension and ls_id prefix
            name_part = radio_basename.replace('.fits', '').replace(ls_id_query_str + '_', '')
            
            # Extract survey type (RACS-High, RACS-Mid, RACS-Low) and index
            if '_' in name_part:
                parts = name_part.split('_')
                survey_type = parts[0]  # e.g., RACS-High
                index = parts[1] if len(parts) > 1 else '0'
            else:
                survey_type = name_part
                index = '0'
            
            # Convert survey type to lowercase and remove RACS- prefix for filename
            survey_short = survey_type.replace('RACS-', '').lower()
            
            # Generate output filename
            output_filename = os.path.join(output_dir, "{0}_overlay_{1}{2}.png".format(ls_id_query_str, survey_short, index))
            
            print("\nProcessing: {0}".format(radio_basename))
            print("  Creating overlay: {0}".format(output_filename))
            
            # Determine optical label from filename
            optical_label = get_optical_label_from_filename(optical_file_path)
            
            # Create overlay for this radio file
            try:
                create_radio_optical_overlay([radio_file], [optical_file_path], [coords_source[0]], 
                                             output_filename, image_size_asec=args.image_size, 
                                             optical_label=optical_label, ls_id=ls_id_query_str)
                print("  Saved: {0}".format(output_filename))
                successful += 1
            except Exception as e:
                print("  Error creating overlay: {0}".format(e))
                failed += 1
        
        # Mark this ls_id as processed if at least one overlay was created
        if len(radio_files) > 0 and successful > 0:
            print("\nCompleted processing ls_id: {0}".format(ls_id_query_str))
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print("Total ls_ids to process: {0}".format(len(ls_ids_to_process)))
    print("Successfully processed: {0}".format(successful))
    print("Failed: {0}".format(failed))
    print("Skipped (missing files/folders): {0}".format(skipped))
    print("=" * 60)

