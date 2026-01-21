#!/usr/bin/env python
"""
Plot quasar spectra from .dat files in confirmed_Mkrtchyan folder.
Creates 3 subplots arranged vertically (stacked on top of each other).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii

# Files to plot with object names and redshifts (ordered by redshift: smallest to largest)
files = [
    ('/Users/dathev/PhDProjects/observations/confirmed_Mkrtchyan/mkrt0135m0552.dat', 'J0135-0552', 5.000),
    ('/Users/dathev/PhDProjects/observations/confirmed_Mkrtchyan/mkrtchyan0236m1147.dat', 'J0236-1147', 5.177),
    ('/Users/dathev/PhDProjects/observations/confirmed_Mkrtchyan/hizJ342m18.dat', 'J2248-1803', 5.606)
]

# Rest-frame wavelengths of common emission lines (in Angstroms)
emission_lines = {
    'Lya': 1215.67
}

def read_spectrum(filename):
    """Read spectrum from .dat file. Returns wavelength and flux arrays."""
    data = ascii.read(filename)
    wave = np.array(data[data.colnames[0]])
    flux = np.array(data[data.colnames[1]])
    return wave, flux

def calculate_emission_line_positions(z, rest_wavelengths):
    """Calculate observed wavelengths of emission lines given redshift."""
    return {name: wave * (1 + z) for name, wave in rest_wavelengths.items()}

def plot_spectrum(ax, wave, flux, line_positions, obj_name, z):
    """Plot a single spectrum on the given axes."""
    # Scale flux to units of 10^-18 erg/s/cm²/Å for display
    flux_scaled = flux * 1e18
    ax.plot(wave, flux_scaled, 'k-', linewidth=1.2, alpha=0.9, zorder=2)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    
    # Add emission line markers
    for line_name, obs_wave in line_positions.items():
        if obs_wave >= wave.min() and obs_wave <= wave.max():
            ax.axvline(x=obs_wave, color='k', linestyle='--', linewidth=2.0, alpha=0.6, zorder=3)
    
    # Remove individual axis labels - will add common ones later
    
    wave_range = wave.max() - wave.min()
    ax.set_xlim(wave.min() - 0.01*wave_range, wave.max() + 0.01*wave_range)
    
    flux_valid = flux_scaled[~np.isnan(flux_scaled) & np.isfinite(flux_scaled)]
    flux_max = np.nanmax(flux_valid)
    flux_min = np.nanmin(flux_valid)
    flux_range = flux_max - flux_min
    ax.set_ylim(max(flux_min - 0.2*flux_range, flux_min - 5), flux_max + 0.3*flux_range)
    
    name_str = f"{obj_name}, z={z:.3f}"
    ax.text(0.02, 0.95, name_str, transform=ax.transAxes,
           ha='left', va='top', fontsize=20, weight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    
    # Format y-axis ticks to show reasonable values (no scientific notation)
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    
    return line_positions

def main():
    """Main function to create the 3-panel spectrum plot."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, left=0.15, right=0.98, top=0.92, bottom=0.12)
    
    for idx, (filename, obj_name, z) in enumerate(files):
        wave, flux = read_spectrum(filename)
        line_positions = calculate_emission_line_positions(z, emission_lines)
        plot_spectrum(axes[idx], wave, flux, line_positions, obj_name, z)
        
        # Zoom in on y-axis and set x-axis limit for top and bottom subplots
        if idx == 0 or idx == 2:  # top (idx=0) and bottom (idx=2)
            # Set x-axis maximum to 10000 Angstroms
            xlim = axes[idx].get_xlim()
            axes[idx].set_xlim(xlim[0], 10000)
            
            # Find flux values around Lya emission line (scale to 10^-18 units)
            flux_scaled = flux * 1e18
            for line_name, obs_wave in line_positions.items():
                if obs_wave >= wave.min() and obs_wave <= wave.max():
                    # Define a window around Lya (e.g., ±500 Angstroms)
                    window = 500
                    mask = (wave >= obs_wave - window) & (wave <= obs_wave + window)
                    flux_around_lya = flux_scaled[mask]
                    flux_around_lya_valid = flux_around_lya[~np.isnan(flux_around_lya) & np.isfinite(flux_around_lya)]
                    
                    if len(flux_around_lya_valid) > 0:
                        flux_min_lya = np.nanmin(flux_around_lya_valid)
                        flux_max_lya = np.nanmax(flux_around_lya_valid)
                        flux_range_lya = flux_max_lya - flux_min_lya
                        # Set tighter y-axis range focused on Lya region
                        ymin = max(flux_min_lya - 0.3*flux_range_lya, np.nanmin(flux_scaled[~np.isnan(flux_scaled) & np.isfinite(flux_scaled)]) - 0.5)
                        ymax = flux_max_lya + 0.4*flux_range_lya
                        axes[idx].set_ylim(ymin, ymax)
                    break
        
        # Add Lya label at the top of each subplot
        for line_name, obs_wave in line_positions.items():
            if obs_wave >= wave.min() and obs_wave <= wave.max():
                xlim = axes[idx].get_xlim()
                x_frac = (obs_wave - xlim[0]) / (xlim[1] - xlim[0])
                # Position label consistently for all plots
                y_pos = 1.05
                axes[idx].text(x_frac, y_pos, line_name, transform=axes[idx].transAxes,
                              ha='center', va='bottom', fontsize=16, weight='normal')
    
    # Add common y-axis label on the left side with 10^-18
    fig.text(0.06, 0.5, r'F$_\lambda$ [10$^{-18}$ erg/s/cm²/Å]', 
            rotation=90, ha='center', va='center', fontsize=20, weight='normal')
    
    # Add common x-axis label at the bottom
    fig.text(0.5, 0.04, 'λ Obs. [Å]', 
            ha='center', va='bottom', fontsize=20, weight='normal')
    
    plt.savefig('quasar_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
