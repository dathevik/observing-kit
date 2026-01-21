from __future__ import division, print_function
import math
import sys

import numpy as np
#constants
from astropy import constants as const
import astropy.units as u
c = 2.99792458e10 #cm/s
Msun = 1.989e33 # g

#restframe wavelengths
lya0 =  1215.668 #Angstroms
nv0 = 1240.0 #Angstrom

#Unit conversions
jansky2cgs = 10 ** (-23)
angstrom2cm = 10 ** (-8)
Mpc2cm = 3.08568025e24

#Molecular transition frequencies
#More here:http://spec.jpl.nasa.gov/ftp/pub/catalog/catform.html
CO76_f0 = 806.6518060#GHz
CI21_f0 = 809.3435000#GHz
CO54_f0 = 576.2679305# GHz
CO65_f0 = 691.4730763# GHz
CO43_f0 = 461.0407682#GHz
CO32_f0 = 345.7959899#GHz
CO21_f0 = 230.5380000#GHz
CO10_f0 = 115.2712018#GHz

CII_f0 = 1900.539#GHz

COtrans = {'10':CO10_f0,
           '21':CO21_f0,
           '32':CO32_f0,
           '43':CO43_f0,
           '54':CO54_f0,
           '65':CO65_f0,
           '76':CO76_f0}

effective_wavelengths = {
                        "yps1":9627.7,
                        "Y_ukirt":10289.4, #http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=UKIRT/UKIDSS.Y
                        "J_ukirt":12444.0,
                        "J_grond":12377.6,
                        "J_sofi":12470.,
                        "H_ukirt":16221.2,


                        }


ebv_factor = {
            "decam":{"g":3.237, "r":2.176, "z":1.217},
            "ps1":{"g":3.172, "r":2.271, "i":1.682,
                    "z":1.322, "y":1.087}
            }

vega2ab = {
            "wise":{"w1":2.699, "w2":3.339, "w3":5.174, "w4":6.620},
            "2mass":{"j":0.89, "h":1.37, "k":1.84},
            "ukirt":{"z":0.528, "y":0.634, "j":0.938, "h":1.379, "k":1.9}

            }
           

# RA(radians),Dec(radians),distance(kpc) of Galactic center in J2000
Galactic_Center_Equatorial=(math.radians(266.40510), math.radians(-28.936175), 8.33)

# RA(radians),Dec(radians) of Galactic Northpole in J2000
Galactic_Northpole_Equatorial=(math.radians(192.859508), math.radians(27.128336))


def angle_dist():
    '''
    Calculate the angular distance and the sky angle between two objects
    '''
    ra1 = input("RA1:")
    dec1 = input("DEC1:")
    ra2 = input("RA2:")
    dec2 = input("DEC2:")
    
    dra = ra2 - ra1
    ddec = dec2 - dec1
    
    print("RA_OFFSET: {0} degrees" .format(dra))
    print("RA_OFFSET: {0} arcsecs" .format(dra * 3600.))
    print("DEC_OFFSET: {0} degrees" .format(ddec))
    print("DEC_OFFSET: {0} arcsecs" .format(ddec * 3600.))
    
    dist = calc_distance(ra1, dec1, ra2, dec2)
    
    print("DISTANCE: {0} degrees" .format(dist))
    print("DISTANCE: {0} arcsecs" .format(dist * 3600.))
    
    calc_angle(dist, dra, ddec)
    
    
def apparent_mag(absolute_mag, redshift, cosmology):
    '''
    Calculates the apparent magnitude for an object with known
    absolute magnitude and redshift for a given cosmology.
    
    Input
        absolute_mag: Absolute magnitude
        redshift: Source redshift
        cosmology: an astropy cosmology object with the desired cosmology
           e.g. from astropy.cosmology import WMAP9 or
           from astropy import cosmology
           cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.28)
           Planck15 = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.307)
    '''
    #Distance modulus + K-correction
    DM = cosmology.distmod(redshift).value - 2.5 * np.log10(1+redshift)
    m = DM + absolute_mag
    
    return m
    
    
def absolute_mag(apparent_mag, redshift, cosmology):
    '''
    Calculates the Absolute magnitude for an object with known
    apparent magnitude and redshift for a given cosmology.
    
    Input
        apparent_mag: Apparent magnitude
        redshift: Source redshift
        cosmology: an astropy cosmology object with the desired cosmology
           e.g. from astropy.cosmology import WMAP9 or
           from astropy import cosmology
           cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.28)
           Planck15 = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.307)
    '''
    #Distance modulus + K-correction
    DM = cosmology.distmod(redshift).value - 2.5 * np.log10(1+redshift)
    M = apparent_mag - DM
    
    return M


def between(a, vmin, vmax):
    """ Return a boolean array True where vmin <= a < vmax.
    Notes
    -----
    Careful of floating point issues when dealing with equalities.
    """
    a = np.asarray(a)
    c = a < vmax
    c &= a >= vmin
    return c
    
    


    
def gcirc(ra1, dc1, ra2, dc2, verbose=True):
    '''
    Calculate the rigorous great circle arc distances.
    
        INPUT:
        RA1, DEC1 coordinates of point 1
        RA2, DEC2 coordinates of point 2
        coordinates in sexagesimal
        
        OUTPUT:
            DIS: Anglular distance on the sky between points 1 and 2 in arcsec

        PROCEDURE:
    "Cosine formula" (p.12 of Green's Spherical Astronomy
      
       TODO: 
        make it more similar to 
       See: http://idlastro.gsfc.nasa.gov/ftp/pro/astro/gcirc.pro
             
    '''
    rarad1 = np.radians(ra1)
    dcrad1 = np.radians(dc1)
    rarad2 = np.radians(ra2)
    dcrad2 = np.radians(dc2)

    radif = np.abs(rarad1 - rarad2)
    
    costt1 = math.sin(dcrad1) * np.sin(dcrad2) 
    costt2 = math.cos(dcrad1) * np.cos(dcrad2) * np.cos(radif)
    cosdis = costt1 + costt2
    
    dis = np.arccos(cosdis)
    
    disas = np.degrees(dis) * 3600.
    
    if verbose:
        print("Angular separation is {0} arcsec" .format(disas))
        
    return disas
    

def eq2gal(ra,dec):
    """
    Convert Equatorial coordinates to Galactic Coordinates in the epch J2000.
    
    Keywords arguments:
    ra  -- Right Ascension (in degrees)
    dec -- Declination (in degrees)

    Return a tuple (l, b):
    l -- Galactic longitude (in degrees)
    b -- Galactic latitude (in degrees)
    """
    # RA(radians),Dec(radians),distance(kpc) of Galactic center in J2000
    Galactic_Center_Equatorial=(math.radians(266.40510), math.radians(-28.936175), 8.33)

    # RA(radians),Dec(radians) of Galactic Northpole in J2000
    Galactic_Northpole_Equatorial=(math.radians(192.859508), math.radians(27.128336))

    ra = np.radians(ra)
    dec = np.radians(dec)
    alpha = Galactic_Northpole_Equatorial[0]
    delta = Galactic_Northpole_Equatorial[1]
    la = math.radians(33.0)
    
    b = np.arcsin(np.sin(dec) * np.sin(delta) +
                  np.cos(dec) * np.cos(delta) * np.cos(ra - alpha))

    l = np.arctan2(np.sin(dec) * np.cos(delta) - 
                   np.cos(dec) * np.sin(delta) * np.cos(ra - alpha), 
                   np.cos(dec) * np.sin(ra - alpha)
                   ) + la
    if l.ndim == 0:
        l = l if l >= 0 else (l + math.pi * 2.0)
    else:
        l[l < 0] = l[l < 0] + math.pi * 2.0

    l = l % (2.0 * math.pi)

    return np.degrees(l), np.degrees(b)


def equivalent_width(Flya, C, redshifted_lya, beta):
    '''
    Calculate Equivalent width
    INPUT
        Flya
            Lyman Alpha flux [cgs]
        C
            Continuum Strength [cgs]
        redshifted_lya
            Ly_alpha * (1+z) [Angstroms]
        beta
            Continuum powerlaw slope (f_lambda ~ lambda ** beta)
    '''
    EW = Flya / (C * (redshifted_lya * angstrom2cm) ** beta) * (1 / angstrom2cm)
    return EW
    

def flux_density_lambda(mag_ab, lambda_eff):
    '''
    Calculate the flux density in terms of wavelength (f_lambda)
    INPUT
        mag_ab [float]
            magnitude AB  
        lambda_eff [float]
            Effective filter wavelength in Angstroms!  
    RETURN
        Flux density in terms of wavelength in cgs [float]   
        
    TODO
        Check if I should use f_lambda in cm or Angstroms.    
    '''
    lambda_eff = lambda_eff * angstrom2cm
    f_lambda = (c / lambda_eff ** 2) * flux_density_nu(mag_ab, units='cgs') 
    
    return f_lambda
    

def flux_density_nu(mag_ab, mag_ab_error=None, units='cgs', verbose=True):
    '''
    Calculate the flux densitiy in terms of frequency (f_nu)
    INPUT
        mag_ab [float]
            magnitude AB
        mag_ab_error [float]
            error of AB magnitude. None if there is no error known
        units [str]
            Output Units. Choices: 'janksy', 'cgs'
    RETURN
        Flux density in the units specified in units [float]            
    '''
    #Reference: Cosmological Physics. John Peacock. Page 399 chap13.2
    f_nu = 10 ** (0.4 * (8.90 - mag_ab))
    if units=='jansky':
        #return f_nu
        pass
    elif units=='cgs':
        f_nu = f_nu * jansky2cgs
    else: 
        msg='Flux Density units not supported'
        sys.exit(msg) 
    if mag_ab_error is None:
        if verbose:
            print("Flux = ", f_nu)
        return f_nu
    else:
        flux_error = np.abs(-0.04 * mag_ab * f_nu * mag_ab_error)
        if verbose:
            print("Flux = ", f_nu)
            print("Flux Error = ", flux_error)
        return f_nu, flux_error
        
        
def mag_ab(fnu, fnu_error=None, un=u.Jansky, verbose=True):
    '''
    Calculate AB magnitude from fnu
    
    It needs an astropy.unit objects with the units of the fnu. It will
    be transformed to Janskys
    '''
    tojansky = un.to(u.Jansky)
    fnu_jansky = fnu * tojansky
    mag_ab = -2.5 * np.log10(fnu_jansky) + 8.90
    
    if fnu_error is None:
        if verbose:
            print("mag_ab = ", mag_ab)
        return mag_ab
    else:
        cte = -2.5 / np.log(10)
        mag_ab_error = np.abs(cte * fnu_error/fnu)
        if verbose:
            print("mag_ab = ", mag_ab)
            print("mag_ab Error = ", mag_ab_error)
        return mag_ab, mag_ab_error
        

#def flux_density_nu_error(mag_ab, mag_ab_error, units='cgs', )
        
        
def flux_lya_cont(fNB_l, fBB_l, DeltaNB, effw_NB, effw_BB, beta=-2):
    '''
    Calculate the Lyman Alpha Flux and Continuum strength from the measurements
    of one Narrow Band and a Broad Band filters. The Broad Band filter 
    redwards from the Ly alpha line (See Venemans 2005)
    INPUT
        fNB_l [float]
            Flux density in the Narrow band in terms of wavelength [cgs]
        fBB_l [float]
            Flux density in the Broad band in terms of wavelength [cgs]
        DeltaNB [float]
            Narrow band widht [Angstroms!]
        effw_NB [float]
            Effective wavelength of the Narrow Band filter [Angstroms]
        effw_BB [float]
            Effective wavelength of the Broad Band filter [Angstroms]
        beta
            Continuum powerlaw slope (f_lambda ~ lambda ** beta)
            Default beta -2 means flat continuum           
    RETURN
        Lyman Alpha flux and Continuum strenght [cgs] 
    '''
    DeltaNB = DeltaNB * angstrom2cm
    effw_NB = effw_NB * angstrom2cm
    effw_BB = effw_BB * angstrom2cm
    flya = DeltaNB * (fNB_l - fBB_l * (effw_NB / effw_BB) ** beta)
    Cstrength = fBB_l / (effw_BB ** beta)
    return flya, Cstrength
    
    
def get_flux_density(flux_density_observed, wave_or_freq_0,
                        wave_or_freq_observed, alpha_nu, 
                        redshift=0, wave_or_freq='wave',
                        wave_optical_rest=False,
                        flux_density_observed_error=None):
    """
    Estimate a flux density at a wavelength or frequency given by wave_or_freq_0
    based on the observed flux density at the wavelength or frequency in wave_or_freq_observed
    IF wave_optical_rest=True. it is assumed that the
     wave_or_freq_observed is actually in the rest frame.
    
        
    Input:
        
        
        flux_density_observed: observed flux density 
        
        wave_or_freq_0: wavelength or frequency at which the output flux_density will be calculated.
                        It has to have the same units as wave_or_freq_observed. Whether is wave or freq
                        is defined in wave_or_freq.
                        
        wave_or_freq_observed: wavelength or frequency at which the input flux_density was calculated
                        It has to have the same units as wave_or_freq_0. Whether is wave or freq
                        is defined in wave_or_freq.
                        
        alpha_nu: Spectral index (Typical value for radio: -0.75, for optical -0.5)
        
        flux_units: astropy.units units for flux density (input and output)
                    for example: 
                    import astropy.units as u
                    u.microJansky
                    TODO Implement.
        
        redshift: The redshift of the source(s).
                  If redshift=0. It is assumed that the input observed wave or freq
                  are actually at rest-frame
                  
        wave_or_freq: Specify if the inputs are wavelenght or frequency
                accepted: ['wave', 'freq']

        flux_density_observed_error= Optional parameter. The error of 
                                    the observed flux density.
                                     If None is ignored. If given, the 
                                     error is calculated

        Output: 
        
        flux_density: the estimated flux_density
        
        flux_density_error: the estimated error for flux_density 
        (only if flux_density_observed_error is not None
    """
    if wave_or_freq == 'wave':
        C = (wave_or_freq_observed / wave_or_freq_0) ** alpha_nu
    elif wave_or_freq == 'freq':
        C = (wave_or_freq_observed / wave_or_freq_0) ** -alpha_nu
    else:
        raise ValueError('wave_or_freq: {:0} not supported'.format(wave_or_freq))
        
    flux_density = flux_density_observed * C
    
    if not wave_optical_rest:
        print("CHECK")
        flux_density *= (1 + redshift) ** -alpha_nu
        
    if flux_density_observed_error is None:
        return flux_density
    else:
       #only where the erros is different to -1
        mask = flux_density_observed_error != -1
        flux_density_error = np.ones_like(flux_density_observed_error) * -1
        try:
            flux_density_error[mask] = flux_density_observed_error[mask] * C
        except:
            print("Something wrong line 433 astrotools.py")
            flux_density_error = flux_density_observed_error * C
            print(flux_density_error)
        if not wave_optical_rest:
            flux_density_error[mask] *= (1 + redshift[mask]) ** -alpha_nu
        return flux_density, flux_density_error    
       
  
def get_ind(val, arr):
    '''
    get a value and an array.
    Returns the index of the closest element of the array to the value
    '''
    return min(range(len(arr)), key=lambda i: abs(arr[i] - val))
  
    
def get_pixelscale(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''
    if 'CDELT1' in hdr.keys():
        CD1 = hdr['CDELT1']
        CD2 = hdr['CDELT2']
    elif 'CD1_1' in hdr.keys():
        CD1 = hdr['CD1_1']
        CD2 = hdr['CD2_2']
    else:
        print('pixel scale unknown. Using 1 pix/arcsec')
        CD1 = CD2 = 1
    
    scale = 0.5 * (np.abs(CD1) + np.abs(CD2)) * 3600
    
    return scale
    
#probably delete this
def get_pixelscale_v1(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''
    if hdr.has_key('CDELT1'):
        CD1 = hdr['CDELT1']
        CD2 = hdr['CDELT2']
    elif hdr.has_key('CD1_1'):
        CD1 = hdr['CD1_1']
        CD2 = hdr['CD2_2']
    else:
        print('pixel scale unknown. Using 1 pix/arcsec')
        CD1 = CD2 = 1
    
    scale = 0.5 * (np.abs(CD1) + np.abs(CD2)) * 3600
    
    return scale   
 
    
def luminosity_lyalpha(flux_lyalpha, luminosity_distance):
    '''
    Calculate the Lyman alpha luminosity
    INPUT
        flux_lyalpha [float]
            Lyman Alpha Flux [cgs]
        luminosity_distance [float]
            Luminosity distance at the required redshift [Mpc]
    RETURN
        Lyman alpha luminosity [erg/s]
        
    '''
    luminosity_distance = luminosity_distance * Mpc2cm
    return 4 * np.pi * luminosity_distance ** 2 * flux_lyalpha
    
    
def freq2lambda(freq, fin='GHz', lout='micron'):
    ''' Receives freq in GHz. Returns lambda in microns
        So far only tested for GHz->micron. in developing
    '''
    return c / freq
    
    
def sfr_Halpha(lyaL):
    '''
    Determine the Star formation rate (SFR) assuming a ratio between
    Lyman alpha luminosity (lyaL) and H alpha luminosity of 8.7 (case B
    recombination/approximation? (Brocklehurst 1971)) and a H alpha luminosity
    to SFR conversion for a Salpeter initail mass function from Madau 1998
    (See Venemans 2005 and Ouchi 2008)
    
    INPUT 
        lyaL
            Lyman alpha luminosity in cgs
    RETURN
        SFR in Solar Masses per year
    '''
    return lyaL / 1.1e42
    

def _interp(x, xp, fp, left=None, right=None):
    """
    Taken from: https://github.com/keflavich/pyspeckit/blob/master/pyspeckit/spectrum/interpolation.py
    Overrides numpy's interp function, which fails to check for increasingness....
    """
    indices = np.argsort(xp)
    xp = np.array(xp)[indices]
    fp = np.array(fp)[indices]
    return np.interp(x, xp, fp, left, right)

    
def smooth(data, smooth, smoothtype='gaussian', downsample=True,
           downsample_factor=None, convmode='same'):
    """
    Taken from https://github.com/keflavich/pyspeckit/blob/master/pyspeckit/spectrum/smooth.py


    Smooth and downsample the data array.  NaN data points will be replaced
    with interpolated values
    Parameters
    ----------
    smooth  :  float 
        Number of pixels to smooth by
    smoothtype : [ 'gaussian','hanning', or 'boxcar' ]
        type of smoothing kernel to use
    downsample :  bool 
        Downsample the data?
    downsample_factor  :  int 
        Downsample by the smoothing factor, or something else?
    convmode : [ 'full','valid','same' ]
        see :mod:`numpy.convolve`.  'same' returns an array of the same length as
        'data' (assuming data is larger than the kernel)
    """
    
    roundsmooth = round(smooth) # can only downsample by integers

    if downsample_factor is None and downsample:
        downsample_factor = int(roundsmooth)
    elif downsample_factor is None:
        downsample_factor = 1

    if smooth > len(data) or downsample_factor > len(data):
        raise ValueError("Error: trying to smooth by more than the spectral length.")

    if smoothtype == 'hanning':
        kernel = np.hanning(2+roundsmooth)/np.hanning(2+roundsmooth).sum()
    elif smoothtype == 'gaussian':
        xkern  = np.linspace(-5*smooth,5*smooth,smooth*11)
        kernel = np.exp(-xkern**2/(2*(smooth/np.sqrt(8*np.log(2)))**2))
        kernel /= kernel.sum()
        if len(kernel) > len(data):
            lengthdiff = len(kernel)-len(data)
            if lengthdiff % 2 == 0: # make kernel same size as data
                kernel = kernel[lengthdiff/2:-lengthdiff/2]
            else: # make kernel 1 pixel smaller than data but still symmetric
                kernel = kernel[lengthdiff/2+1:-lengthdiff/2-1]
    elif smoothtype == 'boxcar':
        kernel = np.ones(roundsmooth)/float(roundsmooth)

    # deal with NANs or masked values
    if hasattr(data,'mask'):
        if type(data.mask) is np.ndarray:
            OK = True - data.mask
            if OK.sum() > 0:
                data = interpolation._interp(np.arange(len(data)),np.arange(len(data))[OK],data[OK])
            else:
                data = OK
    if np.any(True - np.isfinite(data)):
        OK = np.isfinite(data)
        if OK.sum() > 0:
            data = _interp(np.arange(len(data)),np.arange(len(data))[OK],data[OK])
        else:
            data = OK

    if np.any(True - np.isfinite(data)):
        raise ValueError("NANs in data array after they have been forcibly removed.")

    smdata = np.convolve(data,kernel,convmode)[::downsample_factor]

    return smdata
    
    
def get_lyalum_ew_sfr(magNB, magBB, effwNB, DeltaNB, effwBB, DL, beta=-2):
    '''
    Handy function to get Lyman Alpha luminosities in erg/s, Observed equivalent
    widths in Angstroms and SFR(Halpha) in Solar Masses per year
    
    DL [float]
            Luminosity distance at the required redshift [Mpc]
    '''
    flux_densityNB = flux_density_lambda(magNB, effwNB)
    flux_densityBB = flux_density_lambda(magBB, effwBB)

    Flya, C = flux_lya_cont(flux_densityNB, flux_densityBB, DeltaNB, effwNB,
                                                                 effwBB, beta)
                                                                 
    lumlya = luminosity_lyalpha(Flya, DL)
    ew = equivalent_width(Flya, C, effwNB, beta)
    sfr = sfr_Halpha(lumlya)
    
    return lumlya, ew, sfr
    
    
def luminosity_specific(flux_density, redshift, cosmology,
                                     flux_density_units):
    """
    Calculates the specific luminosity given by
     Lnu = 4pi LuminosityDistance^2 * flux_density / (1+z)
    
    flux_density
           The flux density to transofr into specific luminosity
           
    redshift
          The redshift of the sources
           
    cosmology
          Cosmology to be used for the luminosity distance calculation
           e.g. from astropy.cosmology import WMAP9 or
           from astropy import cosmology
           cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.28)
           
    flux_density_units
           astropy units of the input flux_density
           
     Return
        specific_luminosity in W/Hz
    """
    flux_density = flux_density #* flux_density_units
    #print(flux_density.data)
    LD = cosmology.luminosity_distance(redshift)
    C =4 * np.pi
    specific_luminosity = C * (LD ** 2) * flux_density * flux_density_units
    specific_luminosity /= (1 + redshift)
    #print(specific_luminosity)
    
    return specific_luminosity.to(u.Watt / u.Hz)
    
    
def fnu_from_Lnu(Lnu, redshift, cosmology,
                                     Lnu_density_units):
    """
    Calculates the specific flux density luminosity given by
     fnu = Lnu / (4pi LuminosityDistance^2) * (1+z)
    
    Lnu
           The specific luminosity
           
    redshift
          The redshift of the sources
           
    cosmology
          Cosmology to be used for the luminosity distance calculation
           e.g. from astropy.cosmology import WMAP9 or
           from astropy import cosmology
           cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.28)
           
    Lnu_density_units
           astropy units of the input Lnu_density
           from astropy import units as u
           u.erg / u.second / u.Hz or u.Watt / u.Hz
           
     Return
        f_nu in Janskys
    """
    
    LD = cosmology.luminosity_distance(redshift)
    C =4 * np.pi
    fnu = Lnu / (C * (LD ** 2)) * Lnu_density_units
    fnu *= (1 + redshift)
    #print(specific_luminosity)
    
    return fnu.to(u.jansky)
    
    
def line_luminosity(line_flux, nu0, z, cosmo, line_flux_error=None):
    '''
    Calculate the line luminosities in solar luminosities as 
    in Section 2.4 Carilli&Walter2013.
    
    It received the integrated line flux in Jansky km/s,
    the rest-frame frequency of the observed line in GHz
    redshift of the source
    and a astropy cosmology which is used to determine the luminosity distance.
    
    '''
    cte = 1.04e-3
    nuobs = nu0 / (1.0 + z)
    DL = cosmo.luminosity_distance(z) #in Mpc
    
    factor= cte * (DL ** 2) * nuobs
    
    lline = line_flux * factor
    
    if line_flux_error is not None:
        lline_error = line_flux_error * factor
        return lline, lline_error
    else:
        return lline
    
    
    
def mag_err(noise_flux_ratio, verbose=True):
    '''
    Calculates the magnitude error from the input noise_flux_ratio
    which is basically the inverse of the Signal-to-Noise ratio
    '''
    err = (2.5 / np.log(10)) * noise_flux_ratio
    if verbose:
        print(err)
    return err
    
    
def mag_from_X_to_Y_sigma(mag, X, Y):
    '''
    Receives a magnitude (mag) that correspond to a
    X sigma-limiting magnitude and returns a Y sigma limiting magnitude
    '''
    '''
    flux = (1. / X) * 10 ** (-0.4 * mag)
    mag_Y = -2.5 * np.log10(Y * flux)
    '''
    mag_Y = mag + 2.5 * np.log10(X / Y)
    return mag_Y
    
    

def nmgy2abmag(flux, flux_ivar=None):
    """
    Conversion from nanomaggies to AB mag as used in the DECALS survey
    flux_ivar= Inverse variance oF DECAM_FLUX (1/nanomaggies^2)
    """
    lenf = len(flux)
    if lenf > 1:
        ii = np.where(flux>0)
        mag = 99.99 + np.zeros_like(flux)
        mag[ii] = 22.5 - 2.5*np.log10(flux[ii])
    else:
        mag = 22.5 - 2.5*np.log10(flux)
    
    if flux_ivar is None:
        return mag
    elif lenf>1:
        err = np.zeros_like(mag)
        df = np.sqrt(1./flux_ivar)
        err[ii] = mag_err(df[ii]/flux[ii], verbose=False)
    else:    
        df = np.sqrt(1./flux_ivar)
        err = mag_err(df/flux, verbose=False)
    
    return mag,err



def abmag2nmgy(m):
    return 10**(-0.4*(m - 22.5))


def posang(ra1, dc1, ra2, dc2, verbose=True):
    '''
    Calculate the rigorous position angle of source 2 (with given RA, Dec) 
      using source 1 (with given RA, Dec) as the center.
    
    INPUT:
        RA1, DEC1 coordinates of point 1
        RA2, DEC2 coordinates of point 2
        coordinates in degrees
        


    OUTPUT:
        ANGLE-- Angle of the great circle containing [ra2, dc2] from
            the meridian containing [ra1, dc1], in the sense north
            through east rotating about [ra1, dc1].  See U above 
            for units.
        RA_OFFSET in arsecs
        DEC_OFFSET in arsecs
    
    PROCEDURE:
        The "four-parts formula" from p.12 of Green's Spherical Astronomy

    BASED on
             see http://idlastro.gsfc.nasa.gov/ftp/pro/astro/posang.pro
    
    EXAMPLE

    For the star 56 Per, the Hipparcos catalog gives a position of 
    RA1 = 66.15593384
    Dec1 = 33.95988843 
    for component A, and 
    RA2 = 66.15646079
    Dec2 =  33.96100069
    for component B.
    What is the position angle of B relative to A?

    astrotools.posang(RA1, Dec1, RA2, Dec2)
    RA offset: 1.57343282843 arcsecs
    DEC offset: 4.004136 arcsecs
    Angle: 21.4522920309 degrees
    '''
    ra_offset = (ra2 - ra1) * np.cos(0.5 * np.radians(dc1 + dc2)) * 3600 
    dec_offset = (dc2 - dc1) * 3600

    rarad1 = np.radians(ra1)
    dcrad1 = np.radians(dc1)
    rarad2 = np.radians(ra2)
    dcrad2 = np.radians(dc2)

    radif = np.abs(rarad1 - rarad2)

    tanang_n = np.sin(radif)
    tanang_d = np.cos(dcrad1) * np.tan(dcrad2) - np.sin(dcrad1) * np.cos(radif) 
    tanang = tanang_n / tanang_d
    angle = np.degrees(np.arctan(tanang))
    
    if verbose:
        print("RA offset: {0} arcsecs" .format(ra_offset))
        print("DEC offset: {0} arcsecs" .format(dec_offset))
        print("Angle: {0} degrees" .format(angle))
    
    return ra_offset, dec_offset, angle      


def read_fire_spectrum(file):
    '''
    get the FIRE spectrum filename of the flux array (*_F_F.fits, returns the wavelenght and flux arrays
    '''
    from astropy.io import fits
    flux, hdr = fits.getdata(file, header=True)
    wave = 10. ** (hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1'])
    return wave, flux
    

    
def radio_loud_parameter(flux_radio, flux_optical, redshift, wave_opt_obs, 
                        freq_rad_obs, freq_rad_0=5., wave_opt_0=4400.,
                        alpha_radio = -0.75, alpha_nu_optical = -0.5,
                        wave_optical_rest=False):
    """
    Estimate the R parameter R=fradio/foptical. Usually 
        R=f5ghz/f4400.
        
    Input:
        flux_radio: the observed flux density (same units as flux_optical)
        
        flux_optical: observed flux density (same units as flux_radio)
        
        redshift: The redshift of the source(s)
        
        wave_opt_obs: Effective wavelength (in Angstroms) of the optical observation
                      (e.g., 33680 (Angstroms) for W1)
                 
        freq_rad_obs: Effective observed frequency for the radio observation (in GHz)
        
        freq_rad_0: Rest-frame frequency (Usually 5 Ghz)
        
        wave_opt_0: Rest-frame optical wavelength (Usually 4400 Angstroms)
        
        alpha_radio: Radio spectral index (good value is -0.75)
        
        alpha_nu_optical: Optical spectral index (-0.5 is reasonably)
        
        wave_optical_rest: True if wave_optical corresponds to rest-frame wavelength. (e.g., W1)
                           False if wave_optical corresponds to observed-frame wavelength (e.g., m1450)
        
        Output: The R parameters
    """
    
    C1 = (wave_opt_obs / wave_opt_0) ** alpha_nu_optical
    C2 = (freq_rad_obs /freq_rad_0) ** alpha_radio
    C = C1 * C2
    
    if wave_optical_rest:
        alpha = - alpha_radio
    else:
        alpha = alpha_nu_optical - alpha_radio

    
    R = (flux_radio/flux_optical) * (1 + redshift) ** alpha / C
    
    return R
    
    
def redshift_from_obs_velocity(rest_freq, vel, vel_err, ref_freq):
    """
    Calculates the redshift and its error for a source where the
    peak did not coincide with the tuning frequency.
    
    Input
        rest_freq
            Float in GHz. The rest-frame frequency of the line
            which the redshift is measured from
        vel
            Float in km/s. The Velocity of the line with respect
            the reference frequency
        vel_err
            Float in km/s. The velocity error
        ref_freq
            Float in GHz. The tuning frequency or sky frequency.
            
    Return
        redshift
            Float. The redshift
        
        redshift_error
            Float. The redshift error
    """
    c = const.c.to('km/s').value
    freq_obs = ref_freq / ((vel / c) + 1.0)
    freq_obs_err = freq_obs * vel_err / (vel + c)
    redshift = (rest_freq / freq_obs) - 1.0
    redshift_error = (redshift + 1.0) * freq_obs_err / freq_obs
    
    print("The measured redshift is z={0:f} +- {1:f}". format(
                                        redshift, redshift_error))
    return redshift, redshift_error 
    

def sn_from_mag_err(err, verbose=True):
    '''
    Calculates the S.N from the magnitude error
    which is basically the inverse of the Signal-to-Noise ratio
    '''
    sn = (2.5 / np.log(10)) / err 
    if verbose:
        print(sn)
    return sn

    
def time_on_source(nvis, nant=6):
    '''
    Calculates the time on source for interferometric data for
    observations of 45 secs.
    
    Print into screen the time in seconds and hours
    
    INPUT
        nvis. int.
            Number of visibilities
        nant. int
            Number of antennae. Default=6.
   
    RETURN
        t_sec
            The time on source in seconds
    '''   
    
    t_sec = (45 * nvis * 2) / (nant * (nant - 1))
    t_hrs = t_sec / 3600.
    
    msg = "{0:d} antennnae equivalent time on source is {1:f} seconds". format(
                                            nant, t_sec)
                                            
    msg2 = "{0:d} antennnae equivalent time on source is {1:f} hours". format(
                                            nant, t_hrs)
    print(msg)
    print(msg2)
    
    return t_sec
    
    
def vel_from_ref(freq, refreq):
    '''Receive the Frequency observed in GHz. 
    And the reference frequency in GHz (known as sky frequency).
    Return the velocity with regard to refreq=sky_freq  in km/s'''
    c_kms = c * 10 ** (-5)
    lam_freq = freq2lambda(freq)
    lam_sky_freq = freq2lambda(refreq)
    v = c_kms * (lam_freq - lam_sky_freq) / lam_sky_freq
    print("Velocity regarding the reference frequence is: {0} km/s" .format(v))
    return v


def zscale(image, nSamples=1000, contrast=0.25):
    """
    Note: Taken from the astropy development codes
    TBD: replace with newly added astropy.zscale function.
    This emulates ds9's zscale feature. Returns the suggested minimum and
    maximum values to display.
    Parameters
    ----------
    image : `~numpy.ndarray`
        The image to compute the scaling on.
    nSamples :  int
        How many samples to take when building the histogram.
    contrast : float
        ???
    """

    stride = image.size/nSamples
    samples = image.flatten()[::stride]
    samples.sort()
    chop_size = int(0.10*len(samples))
    subset = samples[chop_size:-chop_size]

    i_midpoint = int(len(subset)/2)
    I_mid = subset[i_midpoint]

    fit = np.polyfit(np.arange(len(subset)) - i_midpoint, subset, 1)
    # fit = [ slope, intercept]

    z1 = I_mid + fit[0]/contrast * (1-i_midpoint)/1.0
    z2 = I_mid + fit[0]/contrast * (len(subset)-i_midpoint)/1.0
    return z1, z2

