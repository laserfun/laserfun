
import numpy as np
from scipy import integrate

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

def _central_difference(func, x0, dx=1e-6, n=1):
    """Calculate the nth derivative of func at x0 using central difference."""
    if n == 1:
        return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
    elif n == 2:
        return (func(x0 + dx) - 2*func(x0) + func(x0 - dx)) / (dx**2)
    elif n == 3:
        # Central difference for 3rd derivative
        # (-f(x+2h) + 2f(x+h) - 2f(x-h) + f(x-2h)) / (2h^3)
        return (-func(x0 + 2*dx) + 2*func(x0 + dx) - 2*func(x0 - dx) + func(x0 - 2*dx)) / (2 * dx**3)
    else:
        raise NotImplementedError("Only derivatives up to order 3 are implemented.")

class TreacyCompressor:
    """ This class calculates the effects of a grating-based pulse compressor,
        as described in 
        E. B. Treacy, "Optical Pulse Compression With Diffraction Gratings",
        IEEE Journal of Quantum Electronics QE5(9), p454 (1969):
        http://dx.doi.org/10.1109/JQE.1969.1076303
        
    It implements eqn 5b from Treacy1969:
    
    .. math::
    
        dt/dw = \\frac{-4 \\pi^2 c b}{w^3 d^2 (1 - (2 \\pi c / (w d) - \\sin(\\gamma))^2)}
                       
    where :math:`\\gamma` is the diffraction angle, :math:`w` is the angular frequency, 
    :math:`d` is the grating ruling period, and :math:`b` is the slant distance between gratings:
    
    .. math::
    
        b = G \\sec(\\gamma - \\theta)
    
    where :math:`G` is the grating separation and :math:`\\theta` is the acute angle between
    indicent and diffracted rays. The grating equation relates the angles:
    
    .. math::
    
        \\sin(\\gamma - \\theta) + \\sin(\\gamma) = m \\lambda / d
    
    More conventionally, the grating equation is cast in terms of the
    incident and diffracted ray angles:
    
    .. math::
    
        \\sin(\\alpha) + \\sin(\\beta) = m \\lambda / d
    
    It makes sense to solve {3} using the grating specifications (e.g. for
    optimum incident angle :math:`\\alpha`) and then derive Treacy's theta and gamma:
    
    .. math::
    
        \\gamma = \\alpha, \\quad \\theta = \\gamma - \\alpha
    
    This code only considers first order diffraction (m=1).
    """

    def __init__(self, lines_per_mm, incident_angle_degrees=None, littrow_wavelength_nm=None):
        """ Initialize the Treacy Compressor.
        
        You must provide either the incident_angle_degrees OR the littrow_wavelength_nm, from 
        which the incident angle will be calculated.
            
        Parameters
        ----------
        lines_per_mm : float
            Ruling density in lines per millimeter.
        incident_angle_degrees : float, optional
            Incident angle in degrees.
        littrow_wavelength_nm : float, optional
            Design wavelength for Littrow configuration. If provided, the
            incident angle will be set to the Littrow angle for this wavelength, 
            assuming first order diffraction (m=1).
            
        Attributes
        ----------
        self.d : float
            Grating period in meters.
        self.g : float
            Incident angle in radians.
        """
        self.d = 1.0e-3 / lines_per_mm
        
        if incident_angle_degrees is not None and littrow_wavelength_nm is not None:
             raise ValueError("Cannot specify both incident_angle_degrees and littrow_wavelength_nm.")
        
        if incident_angle_degrees is not None:
             val = incident_angle_degrees * 2.0*np.pi / 360.0
             self.g = val
        elif littrow_wavelength_nm is not None:
             # Calculate Littrow angle: sin(theta) = lambda / (2*d)
             # m=1 assumed
             l = littrow_wavelength_nm * 1e-9
             arg = l / (2 * self.d)
             if arg > 1.0:
                  raise ValueError(f"Wavelength {littrow_wavelength_nm} nm is too long for grating {lines_per_mm} l/mm (Littrow angle undefined).")
             self.g = np.arcsin(arg)
        else:
             raise ValueError("Must provide either incident_angle_degrees or littrow_wavelength_nm.")
        
    def calc_dispersion(self, wavelength_nm, grating_separation_meters, order=[2, 3, 4]):
        """Calculate arbitrary higher-order dispersion (GDD, TOD, etc.).
        
        Calculates the n-th order derivative of spectral phase with respect to
        angular frequency omega.
        
        Parameters
        ----------
        wavelength_nm : float
            Center wavelength in nanometers.
        grating_separation_meters : float
            Perpendicular separation between the gratings in meters.
        order : int or list of int
            Order of dispersion (e.g., 2 for GDD, 3 for TOD). Must be >= 2.
            If a list, returns an array of dispersion values.
            Default is [2, 3, 4], returning GDD, TOD, and 4th order dispersion.
            
        Returns
        -------
        disp : float or ndarray
            Dispersion value(s) in common laser units (ps^n).
        """
        if isinstance(order, (list, np.ndarray)):
            return np.array([self.calc_dispersion(wavelength_nm, grating_separation_meters, o) for o in order])

        if order < 2:
             raise ValueError("Order must be >= 2.")
        
        if order == 2:
            # Result is in s^2
            res_s2 = 2.0 * self.calc_dt_dw_singlepass(wavelength_nm, grating_separation_meters)
            return res_s2 * (1.0e12)**2 # Convert to ps^2

        # For orders > 2, we take the numerical derivative of GDD (which is already in ps^2) 
        def gdd_from_w(w):
             # w in rad/s
             l_meters = 2 * np.pi * c_mks / w
             return self.calc_dispersion(l_meters * 1e9, grating_separation_meters, order=2)

        w0 = 2 * np.pi * c_mks / (wavelength_nm * 1e-9)
        
        # Use numerical derivative. 
        # Since gdd_from_w returns ps^2 and we derive wrt omega (rad/s),
        # each derivative adds a factor of [s].
        # Result units: [ps^2] * [s]^(order-2) = [ps^2] * [10^12 ps]^(order-2) = ps^order
        n_deriv = order - 2
        res = _central_difference(gdd_from_w, w0, dx=w0*1e-5, n=n_deriv)
        return res * (1.0e12)**n_deriv
    
    def calc_dispersion_D(self, wavelength_nm, grating_separation_meters):
        """Calculate dispersion in engineering units (D and S).
        
        Parameters
        ----------
        wavelength_nm : float
            Center wavelength in nanometers.
        grating_separation_meters : float or ndarray
            Perpendicular separation between the gratings in meters.
            
        Returns
        -------
        D : float or ndarray
            Dispersion parameter in ps/nm.
        S : float or ndarray
            Dispersion slope in ps/nm^2.
        """
        # Calculate beta2 and beta3 in ps^2 and ps^3
        disp = self.calc_dispersion(wavelength_nm, grating_separation_meters, order=[2, 3])
        
        # If input was array-like, disp is (2, N)
        if disp.ndim == 2:
            beta2, beta3 = disp
        else:
            beta2, beta3 = disp
            
        D = beta2_to_D2(wavelength_nm, beta2)
        S = beta3_to_D3(wavelength_nm, beta3, beta2=beta2)
        return D, S

    def calc_theta(self, wavelength_nm):
        # solve the grating equation for the diffracted angle
        arg = wavelength_nm * 1.0e-9/self.d - np.sin(self.g)
        
        # For robustness, clamp values slightly outside [-1, 1] due to float precision
        if np.any(arg < -1) or np.any(arg > 1):
             arg = np.clip(arg, -1.0, 1.0)
        
        alpha = np.arcsin(arg)
        theta = self.g - alpha
        return theta

    def calc_dt_dw_singlepass(self, wavelength_nm,
                              grating_separation_meters):
        """Calculate dt/dw (Eq. 1 in class docstring) for a single pass."""
        G = grating_separation_meters
        l = wavelength_nm * 1.0e-9         
        w = 2.0 * np.pi * c_mks / l        
        theta = self.calc_theta(wavelength_nm)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        
        # Equation 1 from docstring
        num = -4.0 * np.pi**2 * c_mks * b
        denom = w**3 * self.d**2 * (1.0 - (2.0*np.pi*c_mks/(w*self.d) - np.sin(gamma))**2 )
        return num / denom
                
    def calc_dphi_domega(self, omega,
                              grating_separation_meters):        
        """Calculate dphi/dw (Group Delay) for a single pass."""
        wavelength_nm = 1.0e9 * 2.0 * np.pi * c_mks / omega
        G = grating_separation_meters
        theta = self.calc_theta(wavelength_nm)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        p = b * (1.0 + np.cos(theta))
        return p / c_mks
        
    def apply_phase_to_pulse(self, grating_separation_meters, pulse):
        """ Apply grating dispersion to a Pulse instance. This applies the 
        full dispersion, without making the GDD/TOD/FOD approximations.
        
        Phase is computed by a double numerical integration of the Group 
        Delay Dispersion (GDD). The integration is performed outwards from 
        the center frequency to maintain precision at the center wavelength
        while still capturing the weirdness that happens near the "grazing limit"
        Wavelengths beyond the grazing limit are zeroed out as they cannot be 
        physically transmitted.
        """
        w0 = pulse.centerfrequency_THz * 2.0 * np.pi * 1.0e12
        w_grid = pulse.w_THz * 1.0e12
        l_meters = (2.0 * np.pi * c_mks / w_grid)

        # 1. Identify diffraction limit (Horizon)
        # arg = l/d - sin(gamma). Diffraction limit is when arg >= 1
        arg = l_meters / self.d - np.sin(self.g)
        
        # A safety mask to handle numerical singularities at Horizon 
        # and ignore non-physical negative frequencies.
        diffraction_mask = (arg < 1.0 - 1e-12) & (w_grid > 0)
        
        # 2. Calculate GDD (double pass)
        y_gdd = np.zeros_like(w_grid)
        # calc_dt_dw_singlepass returns analytical GDD for single pass
        y_gdd[diffraction_mask] = 2.0 * self.calc_dt_dw_singlepass(
            1e9 * l_meters[diffraction_mask], 
            grating_separation_meters
        )
        
        # 3. First integration: GDD -> Relative Group Delay (GD_rel)
        # We integrate from w0 outwards to ensure GD_rel(w0) = 0 and 
        # to maintain 16 orders of precision (avoiding edge singularities).
        w0_idx = np.argmin(np.abs(w_grid - w0))
        gd_rel = np.zeros_like(w_grid)
        
        if w0_idx < len(w_grid) - 1:
            gd_rel[w0_idx:] = integrate.cumulative_trapezoid(
                y_gdd[w0_idx:], w_grid[w0_idx:], initial=0
            )
        if w0_idx > 0:
            gd_rel[:w0_idx+1] = integrate.cumulative_trapezoid(
                y_gdd[:w0_idx+1][::-1], w_grid[:w0_idx+1][::-1], initial=0
            )[::-1]
            
        # 4. Second integration: GD_rel -> Spectral Phase
        phase = np.zeros_like(w_grid)
        if w0_idx < len(w_grid) - 1:
            phase[w0_idx:] = integrate.cumulative_trapezoid(
                gd_rel[w0_idx:], w_grid[w0_idx:], initial=0
            )
        if w0_idx > 0:
            phase[:w0_idx+1] = integrate.cumulative_trapezoid(
                gd_rel[:w0_idx+1][::-1], w_grid[:w0_idx+1][::-1], initial=0
            )[::-1]
            
        # 5. Final shift to ensure phi(w0) = 0 exactly
        phase -= phase[w0_idx]

        # 6. Apply to pulse and mask intensities beyond diffraction limit
        pulse.aw = pulse.aw * np.exp(1j * phase)
        pulse.aw[~diffraction_mask] = 0.0





