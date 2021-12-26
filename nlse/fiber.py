# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate
from scipy.special import factorial
from scipy.optimize import minimize

c_mks = 299792458.0
c_nmps = c_mks * 1e9/1e12

class Fiber:
    """This is a class that contains the information about a fiber."""
    
    def __init__(self, length=0.1, center_wl_nm=1550, dispersion_format='GVD',
                 dispersion=[0], gamma_W_m=0, loss_dB_per_m=0):
        """ 
        Generate a fiber object, which contains the dispersion, nonlinearity, 
        gain, and loss data for a pulse to propagate through. 
                                
        Parameters
        ----------
        length : float
            the length of the fiber in meters
        center_wl_nm : float
            the center wavelength of the fiber in nanometers. This is mainly
            relevant if the dispersion is supplied in terms of the beta 
            coefficients, which expand around a certain wavelength. But, it also
            determines the B values returned by the get_B function.
        dispersion_formats: string 
            determines what format the dispersion is given in. Can be
            'GVD' or 'D' or 'n'
            Corresponding to :
            Beta coefficients (GVD, in units of ps^2/m or ps^2/km) or
            D (ps/nm/km)
            n (effective refractive index)
        dispersion : list or list of arrays
            the format for the dispersion depends on the dispersion_format. 
                - For dispersion_format='GVD', a list of beta coefficients is
                supplied in units of 'ps^n/meter'. Not that the units are per 
                meter and not per kilometer!
                - For dispersion_format='D', a list of two arrays should be 
                 supplied, with the first array being the wavelength 
                 *in nanometers* and the second array being the D value in ps/(nm km).
                - For dispersion_format='n', a list of two arrays should be supplied,
                with the first array being the wavelength *in nanometers* and the 
                second array being the refractive index (n).
        gamma_W_m : float
            This is the nonlinearity in terms of 1/(W m). Note that gamma is 
            often given in units of 1/(W km), and in this case it is necessary
            to set gamma_W_m = 1e-3 * gamma_W_km.
            Gamma is described by Eq. 3.3 in Dudley's book:
            gamma = w0*n2 / (c * Aeff), where w0 is the center frequency, n2 is
            the third-order nonlinearity of the material at w0, c is the speed
            of light, and Aeff is the effective area of the mode at w0.
        loss_dB_per_m : float
            this is the loss expressed as dB per meter
            Note that wavelenght-independent gain can be achieved by using a 
            negative loss. 
        """

        self.length = length
        self.fiberspecs = dict()

        self.center_wavelength = center_wl_nm
        self.gamma = gamma_W_m
        self.alpha = loss_dB_per_m
        
        self.fiberspecs['dispersion_format'] = dispersion_format
        
        if dispersion_format == 'GVD':
            self.betas = np.copy(np.array(dispersion))
            
        elif dispersion_format == 'D':
           raise ValueError('D format is not currently supported')
        
        elif dispersion_format == 'n':
            self.x = dispersion[0]
            self.y = dispersion[1]
        
        else:
           raise ValueError('Dispersion format not recognized.')
        
        self.dispersion_changes_with_z = False
        self.alpha_changes_with_z = False
        self.gamma_changes_with_z = False


    def set_dispersion_function(self, dispersion_function, dispersion_format='GVD'):
        """
        This allows the user to provide a function for the fiber dispersion that can vary as a function
        of `z`, the length along the fiber. The function can either provide beta2, beta3, beta4, etc.
        coefficients, or provide two arrays, wavelength (nm) and D (ps/nm/km)

        Parameters
        ----------
        dispersion_function : function
            A function returning D or Beta coefficients as a function of z.
            z should be in meters.
        dispersion_format : 'GVD' or 'D' or 'n'
            determines if the dispersion will be identified in terms of Beta coefficients
            (GVD, in units of ps^2/m, not ps^2/km) or
            D (ps/nm/km)
            n (effective refractive index)

        Notes
        -----
        For example, this code will create a fiber where Beta2 changes from anomalous
        to zero along the fiber: ::

            Length = 1.5

            def myDispersion(z):

                frac = 1 - z/(Length)

                beta2 = frac * -50e-3
                beta3 = 0
                beta4 = 1e-7

                return beta2, beta3, beta4


        fiber1 = fiber.FiberInstance()
        fiber1.generate_fiber(Length, center_wl_nm=800, betas=myDispersion(0), gamma_W_m=1)


        fiber.set_dispersion_function(myDisperion, dispersion_format='GVD')
        """

        self.dispersion_changes_with_z = True
        self.fiberspecs["dispersion_format"] = dispersion_format
        self.dispersion_function = dispersion_function

    def set_gamma_function(self, gamma_function):
        """
        This allows the user to provide a function for gamma (the effective 
        nonlinearity), in units of 1/(Watts * meters)) that
        can vary as a function of `z`, the length along the fiber.

        Parameters
        ----------
        gamma_function : function
            a function returning gamma as a function of z. z should be in
            units of meters.

        """
        self.gamma_function = gamma_function
        self.gamma_changes_with_z = True
    
    def set_alpha_function(self, gamma_function):
        """
        This allows the user to provide a function for alpha (the loss), 
        in units of 1/Watts that can vary as a function of `z`, the length 
        along the fiber.

        Parameters
        ----------
        alpha_function : function
            a function returning gamma as a function of z. z should be in
            units of meters.

        """
        self.alpha_function = gamma_function
        self.alpha_changes_with_z = True

    def get_gamma(self, z=0):
        """
        Allows the gamma (effective nonlinearity) to be queried at a specific z-position

        Parameters
        ----------
        z : float
            the position along the fiber (in meters)

        Returns
        -------
        gamma : float
            the effective nonlinearity (in units of 1/(Watts * meters))"""

        if self.gamma_changes_with_z:
            gamma = self.gamma_function(z)
        else:
            gamma = self.gamma

        return gamma
    
    
    def get_alpha(self, z=0):
        """
        Allows the alpha (loss per meter) to be queried at a specific z-position

        Parameters
        ----------
        z : float
            the position along the fiber (in meters)

        Returns
        -------
        alpha : float
            the loss (in units of 1/Watts)"""

        if self.alpha_changes_with_z:
            alpha = self.alpha_function(z)
        else:
            alpha = self.alpha

        return alpha

    def get_B(self, pulse, z=0):
        """This provides the propagation constant (beta) at the frequencies of
        the supplied pulse grid. The units are 1/meters.

        Three different methods are used,

        If fiberspecs["dispersion_format"] == "D", then the DTabulationToBetas function is used to
        fit the datapoints in terms of the Beta2, Beta3, etc. coefficients expanded around the pulse
        central frequency.

        If fiberspecs["dispersion_format"] == "GVD", then the betas are calculated as a Taylor expansion
        using the Beta2, Beta3, etc. coefficients around the *fiber* central frequency.
        However, since this expansion is done without the lower order coefficients, the first two
        terms of the Taylor expansion are not defined. In order to provide a nice input for the SSFM,
        which assumes that the group velocity will be zero at the pulse central frequency,
        the slope and offset at the pump central frequency are set to zero.

        If fiberspecs["dispersion_format"] == "n", then the betas are calculated directly from
        the **effective refractive index (n_eff)** as beta = n_eff * 2 * pi / lambda, where lambda is the wavelength
        of the light. In this case, self.x should be the wavelength (in nm) and self.y should be n_eff (unitless).

        Parameters
        ----------
        pulse : an instance of the :class:`pynlo.light.pulse.PulseBase` class
            the pulse must be supplied in order for the frequency grid to be known
        
        z : float
            the postion along the length of the fiber. The units of this must 
            match the units expected by the functions provided to 
            set_dispersion_function() and set_gamma_function(). Should stick 
            to meters.


        Returns
        -------
        B : 1D array of floats
            the propagation constant (beta) at the frequency gridpoints of the supplied pulse
            (units of 1/meters).

        """

        # if the dispersion changes with z, we need to reload the dispersion:
        if self.dispersion_changes_with_z:
            if self.fiberspecs["dispersion_format"] == "D" or self.fiberspecs["dispersion_format"] == "n":
                self.x, self.y = self.dispersion_function(z)
            if self.fiberspecs["dispersion_format"] == "GVD":
                self.betas     = np.array(self.dispersion_function(z))


        B = np.zeros((pulse.npts,))
        if self.fiberspecs["dispersion_format"] == "D":
            raise ValueError('D format is not currently supported')

        elif self.fiberspecs["dispersion_format"] == "GVD":
            # calculate beta[n]/n! * (w-w0)^n
            # w0 is the center of the Taylor expansion, and is defined by the
            # fiber. the w's are from the optical spectrum
            fiber_omega0 =  2*np.pi*c_nmps / self.center_wavelength # THz
            betas = self.betas
            for i in range(len(betas)):
                betas[i] = betas[i]
                B = B + betas[i] / factorial(i + 2) * (pulse.w_THz-fiber_omega0)**(i + 2)

        elif self.fiberspecs["dispersion_format"] == "n":
            # simply interpolate (using a spline) the betas from the refractive index
            # self.x is the wavelength in nm
            # self.y is the refractive index (unitless)

            supplied_W_THz = 2 * np.pi * 1e-12 * 3e8 / (self.x*1e-9)
            supplied_betas = self.y * 2 * np.pi / (self.x * 1e-9)

            # InterpolatedUnivariateSpline wants increasing x, so flip arrays
            interpolator = scipy.interpolate.InterpolatedUnivariateSpline(supplied_W_THz[::-1], supplied_betas[::-1])
            B = interpolator(pulse.w_THz)

        # in the case of "GVD" or "n" it's possible (likely) that the betas will not be zero and have zero
        # slope at the pulse central frequency. For the NLSE, we need to move into a frame propagating at the
        # same group velocity, so we need to set the value and slope of beta at the pulse wavelength to zero:
        if self.fiberspecs["dispersion_format"] == "GVD" or self.fiberspecs["dispersion_format"] == "n":
            center_index = np.argmin(np.abs(pulse.v_THz))
            slope = np.gradient(B)/np.gradient(pulse.w_THz)
            B = B - slope[center_index] * (pulse.v_THz) - B[center_index]

            # print B
            return B

        else:
            return -1



