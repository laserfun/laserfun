
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
        
    def calc_compressor_gdd(self, wavelength_nm, grating_separation_meters):
        """Calculate the Group Delay Dispersion (GDD) [s^2] for a double-pass compressor.
        
        Parameters
        ----------
        wavelength_nm : float
            Center wavelength in nanometers.
        grating_separation_meters : float
            Perpendicular separation between the gratings in meters (G).
            
        Returns
        -------
        gdd : float
            Group Delay Dispersion in s^2.
        """
        return 2.0 * self.calc_dt_dw_singlepass(wavelength_nm,
                                                 grating_separation_meters,
                                                 verbose = False)
    
    def calc_compressor_HOD(self, wavelength_nm, grating_separation_meters, dispersion_order):
        """Calculate arbitrary higher-order dispersion (GDD, TOD, etc.).
        
        Calculates the n-th order derivative of spectral phase with respect to
        angular frequency omega.
        
        Parameters
        ----------
        wavelength_nm : float
            Center wavelength in nanometers.
        grating_separation_meters : float
            Perpendicular separation between the gratings in meters.
        dispersion_order : int
            Order of dispersion (e.g., 2 for GDD, 3 for TOD). Must be >= 2.
            
        Returns
        -------
        hod : float
            Dispersion value in SI units (s^n).
        """
        if dispersion_order < 2:
             raise ValueError("Order must be >= 2.")
        
        if dispersion_order == 2:
            return self.calc_compressor_gdd(wavelength_nm, grating_separation_meters)

        # For TOD (order=3), we want d(GDD)/dw 
        # GDD is 2nd deriv of Phase. TOD is 3rd deriv.
        # calc_compressor_gdd returns GDD.
        # So we need (dispersion_order - 2) derivative of GDD with respect to omega.
        
        def gdd_from_w(w):
             # w in rad/s
             l_meters = 2 * np.pi * c_mks / w
             return self.calc_compressor_gdd(l_meters * 1e9, grating_separation_meters)

        w0 = 2 * np.pi * c_mks / (wavelength_nm * 1e-9)
        
        # Use numerical derivative
        # For TOD (3), n=1 derivative of GDD
        n_deriv = dispersion_order - 2
        return _central_difference(gdd_from_w, w0, dx=w0*1e-5, n=n_deriv)
    
    def calc_theta(self, wavelength_nm):
        l = wavelength_nm * 1.0e-9
        # solve the grating equation for the diffracted angle
        # sin(alpha) + sin(beta) = m lambda / d    (m=1)
        # beta = arcsin(lambda/d - sin(alpha))
        
        arg = l/self.d - np.sin(self.g)
        
        # Check for numeric range issues
        if np.any(arg < -1) or np.any(arg > 1):
             # For robustness, clamp values slightly outside [-1, 1] due to float precision
             arg = np.clip(arg, -1.0, 1.0)
        
        alpha = np.arcsin(arg)
        theta = self.g - alpha
        return theta

    def calc_dt_dw_singlepass(self, wavelength_nm,
                              grating_separation_meters,
                              verbose = False):
        """Calculate dt/dw (Eq. 1 in class docstring) for a single pass."""
        c = c_mks
        G = grating_separation_meters
        l = wavelength_nm * 1.0e-9         
        w = 2.0 * np.pi * c / l        
        theta = self.calc_theta(wavelength_nm)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        
        # Equation 1 from docstring
        num = -4.0 * np.pi**2 * c * b
        denom = w**3 * self.d**2 * (1.0 - (2.0*np.pi*c/(w*self.d) - np.sin(gamma))**2 )
        return num / denom
                
    def calc_dphi_domega(self, omega,
                              grating_separation_meters,
                              verbose = False):        
        """Calculate dphi/dw (Group Delay) for a single pass."""
        c = c_mks
        wavelength_nm = 1.0e9 * 2.0 * np.pi * c / omega
        G = grating_separation_meters
        theta = self.calc_theta(wavelength_nm)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        p = b * (1.0 + np.cos(theta))
        return p / c
        

    
    def apply_phase_to_pulse(self, grating_separation_meters, pulse):
        """ Apply grating dispersion (all orders) to a Pulse instance."""
        w_grid = pulse.w_THz * 2.0 * np.pi * 1e12 # angular frequency grid
        w0 = pulse.centerfrequency_THz * 2.0 * np.pi * 1e12
        
        # Calculate group delay at center frequency to subtract it (center the pulse)
        # dphi/dw represents group delay (τ_g)
        # We integrate from w0, so phase(w0) = 0.
        # But dphi/dw(w0) is not zero. It is the absolute delay through the compressor.
        # We must subtract this linear slope to keep the pulse in the window.
        
        gd_at_center = 2.0 * self.calc_dphi_domega(w0, grating_separation_meters)
        
        def integrand(w):
             # Double pass -> factor of 2.0
             return 2.0 * self.calc_dphi_domega(w, grating_separation_meters)
        
        # Calculate phase by integrating group delay from w0
        # phi(w) = int_{w0}^{w} GD(w') dw'
        # Split integration: forward for w > w0, backward for w < w0
        
        w0_idx = np.argmin(np.abs(w_grid - w0))
        phase = np.zeros_like(w_grid)
        
        # For w >= w0: integrate forward
        if w0_idx < len(w_grid) - 1:
            y_upper = integrand(w_grid[w0_idx:])
            phase_upper = integrate.cumulative_trapezoid(y_upper, w_grid[w0_idx:], initial=0)
            phase[w0_idx:] = phase_upper
        
        # For w < w0: integrate backward (negative integral)
        if w0_idx > 0:
            # Reverse arrays, integrate, then flip and negate
            y_lower = integrand(w_grid[:w0_idx+1][::-1])
            phase_lower = integrate.cumulative_trapezoid(y_lower, w_grid[:w0_idx+1][::-1], initial=0)
            phase[:w0_idx+1] = -phase_lower[::-1]
        
        # Now remove linear component (Group Delay at w0) to keep pulse centered
        # A time delay τ corresponds to phase: phi(w) = -w * τ
        # So to remove the delay, we ADD w * τ to the phase
        # But we want to keep phi(w0) = 0, so we use: w * τ - w0 * τ = (w - w0) * τ
        # Actually, let's think about this more carefully:
        # The group delay GD(w0) represents dphi/dw at w0
        # To remove a constant time shift, we subtract a linear phase
        # But the phase we calculated already has phi(w0) = 0
        # The issue is that dphi/dw at w0 should also be zero for no time shift
        # So we need to subtract: (w - w0) * GD(w0)
        # This makes dphi/dw = 0 at w0 while keeping phi(w0) = 0
        linear_correction = (w_grid - w0) * gd_at_center
        final_phase = phase - linear_correction
        
        # The phase should now have: phi(w0) = 0 and dphi/dw|_{w=w0} = 0
        # But due to grid mismatch, phi(w0_idx) might not be exactly zero
        # Add a constant to ensure phi = 0 at the grid point closest to w0
        final_phase = final_phase - final_phase[w0_idx]
        
        # Apply phase to pulse
        pulse.aw = pulse.aw * np.exp(1j * final_phase)
