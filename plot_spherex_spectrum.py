import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy import constants as const

# Read the SPHEREx spectrum data
data_file = '/Users/dathev/Downloads/357_70884_-12_1290593.dat'

# Read the table using astropy
colnames = ['ra', 'dec', 'x_image', 'y_image', 'mjd', 'flux_bkg', 
            'local_bkg_flg', 'flags', 'fit_ql', 'deep_flg', 'det_id', 
            'lvf_id', 'obs_publisher_did', 'lambda', 'lambda_width', 
            'flux', 'flux_err']
table = ascii.read(data_file, format='tab', comment='#', 
                   names=colnames)

# Extract the last 4 columns: lambda, lambda_width, flux, flux_err
wavelength_um = table['lambda']  # in μm
wavelength_width_um = table['lambda_width']  # in μm
flux_ujy = table['flux']  # in μJy
flux_err_ujy = table['flux_err']  # in μJy

# Convert wavelength from microns to Angstroms
# 1 μm = 10,000 Å
wavelength = wavelength_um * 1e4  # in Å
wavelength_width = wavelength_width_um * 1e4  # in Å

# Convert flux from μJy to CGS units (erg s^-1 cm^-2 Å^-1)
# 1 Jy = 10^-23 erg s^-1 cm^-2 Hz^-1
# F_λ [erg s^-1 cm^-2 Å^-1] = F_ν [erg s^-1 cm^-2 Hz^-1] × (c / λ^2)
# where c is speed of light and λ is in cm
c = const.c.to('cm/s').value  # speed of light in cm/s
wavelength_cm = wavelength_um * 1e-4  # convert μm to cm

# Convert μJy to Jy
flux_jy = flux_ujy * 1e-6  # in Jy
flux_err_jy = flux_err_ujy * 1e-6  # in Jy

# Convert Jy to erg s^-1 cm^-2 Hz^-1
flux_nu = flux_jy * 1e-23  # in erg s^-1 cm^-2 Hz^-1
flux_err_nu = flux_err_jy * 1e-23  # in erg s^-1 cm^-2 Hz^-1

# Convert from per frequency to per wavelength (per Å)
# F_λ = F_ν × (c / λ^2) where λ is in cm, then convert to per Å
flux = flux_nu * (c / wavelength_cm**2) * 1e18  # in erg s^-1 cm^-2 Å^-1
flux_err = flux_err_nu * (c / wavelength_cm**2) * 1e18  # in erg s^-1 cm^-2 Å^-1

# Sort by wavelength for proper line plotting
sort_idx = np.argsort(wavelength)
wavelength = wavelength[sort_idx]
wavelength_width = wavelength_width[sort_idx]
flux = flux[sort_idx]
flux_err = flux_err[sort_idx]

# Redshift for emission line calculations
redshift = 4.9

# Rest-frame wavelengths of emission lines (in Angstroms)
emission_lines_rest = {
    'MgII': 2798.0,  # MgII doublet average
    'Hβ+ [OIII]': 4861.0,  # Hβ + [OIII] average
    'Hα': 6563.0  # Hα
}

# Calculate observed wavelengths of emission lines (in Angstroms)
emission_lines_obs = {name: wave * (1 + redshift) 
                     for name, wave in emission_lines_rest.items()}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12)

# Plot connecting line in dark blue
ax.plot(wavelength, flux, '-', color='#1f77b4', linewidth=1.5, alpha=0.8, zorder=1)

# Plot error bars with black markers and grey error bars
ax.errorbar(wavelength, flux, xerr=wavelength_width, yerr=flux_err,
            fmt='o', color='k', markersize=4, capsize=2, capthick=1.5,
            elinewidth=1.5, ecolor='grey', alpha=0.9, zorder=2)

# Set axis limits first
wave_range = wavelength.max() - wavelength.min()
ax.set_xlim(wavelength.min() - 0.02*wave_range, wavelength.max() + 0.02*wave_range)

# Set y-axis limits
ax.set_ylim(0, 4e9)

# Add vertical dashed lines for spectral features with labels (after limits are set)
for line_name, obs_wave in emission_lines_obs.items():
    if obs_wave >= wavelength.min() and obs_wave <= wavelength.max():
        ax.axvline(x=obs_wave, color='grey', linestyle='--', 
                  linewidth=1.5, alpha=0.7, zorder=0)
        # Add label above the line
        y_pos = 4e9 * 0.95
        ax.text(obs_wave, y_pos, line_name, ha='center', va='bottom',
               fontsize=12, weight='normal', rotation=0)

# Y-axis label
ax.set_ylabel('F$_\\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]', fontsize=14, weight='normal')

# Format axes - no grid, no x-axis label
ax.tick_params(axis='both', labelsize=12)
ax.tick_params(axis='both', which='major', length=6, width=1.5)

plt.tight_layout()

# Show the plot
plt.show()

