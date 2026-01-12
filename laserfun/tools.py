
import numpy as np

c_mks = 299792458.0
c_nmps = c_mks * 1e9 / 1e12

def D2_to_beta2(wavelength_nm, D2):
    """Convert dispersion parameter D [ps/nm/km] to GVD beta2 [ps^2/km]."""
    return -(wavelength_nm**2) / (2.0 * np.pi * c_nmps) * D2

def beta2_to_D2(wavelength_nm, beta2):
    """Convert GVD beta2 [ps^2/km] to dispersion parameter D [ps/nm/km]."""
    return -(2.0 * np.pi * c_nmps) / (wavelength_nm**2) * beta2

def D3_to_beta3(wavelength_nm, D2, D3):
    """Convert dispersion D and slope S to TOD beta3 [ps^3/km]."""
    term_slope = (wavelength_nm**4) / (4.0 * np.pi**2 * c_nmps**2) * D3
    term_D = (wavelength_nm**3) / (2.0 * np.pi**2 * c_nmps**2) * D2
    return term_slope + term_D

def beta3_to_D3(wavelength_nm, beta3, D2=None, beta2=None):
    """Convert TOD beta3 [ps^3/km] to dispersion slope S [ps/nm^2/km].
    
    Requires either D2 [ps/nm/km] or beta2 [ps^2/km].
    """
    if D2 is None:
        if beta2 is None:
            raise ValueError("Provide either D2 or beta2.")
        D2 = beta2_to_D2(wavelength_nm, beta2)

    return (4.0 * np.pi**2 * c_nmps**2 / wavelength_nm**4) * beta3 - (2.0 / wavelength_nm) * D2
