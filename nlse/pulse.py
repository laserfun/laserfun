# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import warnings
import scipy.ndimage.interpolation
import scipy.fftpack as fft

# speed of light in m/s and nm/ps
c_mks = 299792458.0
c_nmps = c_mks * 1e9/1e12

def FFT_t(A,ax=0):
    return fft.ifftshift(fft.ifft(fft.fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))
def IFFT_t(A,ax=0):
    return fft.ifftshift(fft.fft(fft.fftshift(A,axes=(ax,)),axis=ax),axes=(ax,)) 

class Pulse:
    """Class which carried all information about the light field. This class 
       is a base upon which various cases are built, e.g., Gaussian pulses, 
       CW fields, or pulses generated from experimental data.) 
    
       Note that underscores are used for the variables (such as self._npts)
       while self.npts is the getter/setter method.
    """

    def __init__(self, pulse_type='cw', center_wavelength_nm=1550.0, power=1, 
                 fwhm_ps=0.2, time_window_ps=10.0, npts=2**10, frep_MHz=100.,
                 power_is_avg=False, epp=None,
                 GDD=0, TOD=0, chirp2=0, chirp3=0):
        """
        Generate a new pulse object. 
                 
        Parameters
        ----------
        type : string
            sech
                 A(t) = sqrt(P0 [W]) * sech(t/T0 [ps])
                 Note: The full-width-at-half-maximum (FWHM) is given by 
                          T0_ps * 1.76
                
            gaussian
                 Generate Gaussian pulse A(t) = sqrt(peak_power[W]) * 
                         exp( -(t/T0 [ps])^2 / 2 ) centered at wavelength 
                         center_wavelength_nm (nm).
                     Note: For this definition of a Gaussian pulse, T0_ps is the 
                           full-width-at-half-maximum (FWHM) of the pulse.
            sinc
                 Generate sinc pulse A(t) = sqrt(peak_power[W]) * sin(t/T0)/(t/T0)
                     centered at wavelength center_wavelength_nm (nm).
                     The width is given by FWHM_ps, which is the full-width-at-half-maximum 
                     in picoseconds. T0 is equal th FWHM/3.7909885.
                 
        power : float
                 
             
        """
                 
        # the fundamental characteristics of the pulse are:
        #
        # self._npts
        # self._centerfrequency_THz
        # self.time_window_ps
        # self._aw, the complex amplitude grid
        #
        # and, for some pulses:
        # self._power_is_avg
        # self._frep_MHz
        
        # note that we are setting the self._npts variable using the public 
        # self.npts setter method.

        self.npts = npts 
        self.center_wavelength_nm = center_wavelength_nm
        self.time_window_ps = time_window_ps
        
        self.frep_MHz = frep_MHz
        self.power_is_avg = power_is_avg 
        
        T0_ps = fwhm_ps/1.76
        
        if pulse_type == 'sech':                
            # from https://www.rp-photonics.com/sech2_shaped_pulses.html
            self.at = np.sqrt(power)/np.cosh(self.t_ps/(T0_ps))
            
        elif pulse_type == 'gaussian':
            # from https://www.rp-photonics.com/gaussian_pulses.html
            self.at = np.sqrt(power) * np.exp(-2.77*0.5*self.t_ps**2/(T0_ps**2))

        elif pulse_type == 'sinc':
            T0_ps = fwhm_ps/3.7909885   # previously: T0_ps = FWHM_ps/3.7909885
            # numpy.sinc is sin(pi*x)/(pi*x), so we divide by pi
            self.at = np.sqrt(power) * np.sinc(self.t_ps/(T0_ps*np.pi))

        # elif pulse_type == 'cw':
        #     self.at = np.ones(self.npts)
        #     self.aw = self.aw * np.sqrt(power) / sum(abs(self.aw))
        # I'm not convinced this one is handled correctly

        else:
            raise ValueError('Pulse type not recognized.')
                
        if power_is_avg:
            self.at = self.at * np.sqrt( power / ( frep_MHz*1.0e6 * self.epp) )

        if epp is not None:
            self.epp = epp
        
        if pulse_type in ['sech', 'gaussian', 'sinc']:
            self.chirp_pulse_W(GDD, TOD)
            self.chirp_pulse_T(chirp2, chirp3, T0_ps)
        
    # FUNDAMENTAL PROPERTIES
    # These are the fundamental properties that describe the pulse. 
    # All other properties are derived from these
    
    # npts:
    @property
    def npts(self):
        """number of points in the time (and frequency) grid"""
        return self._npts
        
    @npts.setter
    def npts(self, new_npts):
        # TODO: check that npts is reasonable
        if type(new_npts) is int:
            self._npts = new_npts
        else:
            raise ValueError('npts must be an integer.')
    
    # center frequency/wavelength:
    @property
    def centerfrequency_THz(self):
        """ Return center angular frequency (THz) """
        return self._centerfrequency_THz 
        
    @centerfrequency_THz.setter
    def centerfrequency_THz(self, new_centerfrequency):
        assert (new_centerfrequency > 1.0) 
        self._centerfrequency_THz = new_centerfrequency
        
    # time window:
    @property 
    def time_window_ps(self):
        """Gets the time window in picoseconds."""
        return self._time_window_ps
    
    @time_window_ps.setter
    def time_window_ps(self, time_window_ps):
        self._time_window_ps = time_window_ps
    
    # amplitude in frequency domain:
    @property
    def aw(self):
        r""" Get/set the amplitude of the frequency-domain electric field.
        
        Parameters
        ----------
        aw_new : array_like
            New electric field values. 
        
        """
        # need to check that it's the correct dimensions
        return self._aw
        
    @aw.setter           
    def aw(self, aw_new):
        # TODO: need to check that it's the correct dimensions
        if 'self._aw' not in locals()  :
            self._aw = np.zeros((self._npts,), dtype = np.complex128)
        self._aw[:] = aw_new
    
    # frep_MHz:
    @property
    def frep_MHz(self):
        r""" Get/set the pulse repetition frequency. 
        
        This parameter used internally to convert between pulse energy and 
        average power.
        
        Parameters
        ----------
        fr_MHz : float
             New repetition frequency [MHz]
        
        """     
        return self._frep_MHz
    
    @frep_MHz.setter
    def frep_MHz(self, new_frep_MHz):
        if new_frep_MHz > 1.0e6:
            raise ValueError("frep should be specified in MHz. Large value given.")
            
        self._frep_MHz = new_frep_MHz
    
    @property
    def power_is_avg(self):
        """get/set the boolean power_is_avg.
        If True, the power refers to the average power of the laser. 
        if False, the power refers to the peak power of the pulse."""
        return self._power_is_avg
    
    @power_is_avg.setter
    def power_is_avg(self, new_power_is_avg):
        self._power_is_avg = new_power_is_avg
        
        
    # DERIVED PROPERTIES:
    
    # center angular frequency
    @property
    def w0_THz(self):
        """ Return center *angular* frequency (THz) """
        return 2.0 * np.pi * self._centerfrequency_THz 
    
    # center wavelength:
    @property
    def center_wavelength_nm(self):
        return c_nmps / self._centerfrequency_THz
    
    @center_wavelength_nm.setter
    def center_wavelength_nm(self, wl):
        r""" Set the center wavelength of the grid in units of nanometers.
        
        Parameters
        ----------
        wl : float
             New center wavelength [nm]
        
        """
        self.centerfrequency_THz = c_nmps / wl
    
    # frequency grid
    @property
    def v_THz(self):
        """ Return *relative* angular frequency grid (THz)"""
        return 2.0*np.pi*np.arange(-self.npts/2, self.npts/2)/(self.npts*self.dt_ps)
        
    @property
    def w_THz(self):
        """ Return *absolute* angular frequency grid (THz)"""
        return self.v_THz + self.w0_THz
    
    @property
    def f_THz(self):
        """ Return *absolute* angular frequency grid (THz)"""
        return (self.v_THz + self.w0_THz)/(2 * np.pi)
        
    # wavelength grid:
    @property 
    def wavelength_nm(self):
        """ 
        Get the wavelength grid in nanometers.
        
        Returns
        -------
        wl_nm : ndarray, shape npts
            Wavelength grid corresponding to AW [nm]
        """
        return 2*np.pi*c_nmps / self.w
    
    # time grid:
    @property       
    def t_ps(self):
        """ Return temporal grid (ps)"""
        return np.linspace(-self._time_window_ps / 2.0, self._time_window_ps / 2.0,
                         self._npts, endpoint=False)
    # dt:
    @property
    def dt_ps(self):
        """ Return time grid spacing (ps)"""
        return self._time_window_ps / np.double(self._npts)
    
    @property
    def df_THz(self):
        f_THz = self.f_THz
        return f_THz[1] - f_THz[0]
        
    # amplitude in the time domain:
    @property
    def at(self):
        r""" Get/set the amplitude of the time-domain electric field.
        
        Parameters
        ----------
        AW_new : array_like
            New electric field values.
            
        """        
        return IFFT_t( self._aw.copy() )
        
    @at.setter       
    def at(self, at_new):
        self.aw = FFT_t(at_new)
    
    # epp:
    @property
    def epp(self):
        r""" Calculate and return energy per pulse via numerical integration
            of :math:`A^2 dt`
            
            Returns
            -------
            x : float
                Pulse energy [J]
            """
        return (self.dt_ps * 1e-12) * np.trapz(abs(self.at)**2)
    
    @epp.setter
    def epp(self, desired_epp_J):
        r""" Set the energy per pulse (in Joules)
            
            Parameters
            ----------
            desired_epp_J : float
                 the value to set the pulse energy [J]
                 
            Returns
            -------
            nothing
            """
        self.at = self.at * np.sqrt( desired_epp_J / self.epp ) 
        
    def add_noise(self, noise_type='sqrt_N_freq'):
        r""" 
         Adds random intensity and phase noise to a pulse. 
        
        Parameters
        ----------
        noise_type : string
             The method used to add noise. The options are: 
    
             ``sqrt_N_freq`` : which adds noise to each bin in the frequency domain, 
             where the sigma is proportional to sqrt(N), and where N
             is the number of photons in each frequency bin. 
    
             ``one_photon_freq``` : which adds one photon of noise to each frequency bin, regardless of
             the previous value of the electric field in that bin. 
             
        Returns
        -------
        nothing
        """
        
        # This is all to get the number of photons/second in each frequency bin:
        size_of_bins = self.df_THz * 1e-12                 # Bin width in [Hz]
        power_per_bin = np.abs(self.aw)**2 / size_of_bins  # [J*Hz] / [Hz] = [J]
            
        h = 6.62607004e-34 # use scipy's constants package
        
        #photon_energy = h * self.W_THz/(2*np.pi) * 1e12
        photon_energy = h * self.f_THz * 1e-12 # h nu [J]
        photons_per_bin = power_per_bin/photon_energy # photons / second
        photons_per_bin[photons_per_bin<0] = 0 # must be positive.
        
        # now generate some random intensity and phase arrays:
        size = np.shape(self.aw)[0]
        random_intensity = np.random.normal( size=size)
        random_phase     = np.random.uniform(size=size) * 2 * np.pi
        
        if noise_type == 'sqrt_N_freq': # this adds Gausian noise with a sigma=sqrt(photons_per_bin)
                                                                      # [J]         # [Hz]
            noise = random_intensity * np.sqrt(photons_per_bin) * photon_energy * size_of_bins * np.exp(1j*random_phase)
        
        elif noise_type == 'one_photon_freq': # this one photon per bin in the frequecy domain
            noise = random_intensity * photon_energy * size_of_bins * np.exp(1j*random_phase)
        else:
            raise ValueError('noise_type not recognized.')
        
        self.aw = self.aw + noise
        
        
    def chirp_pulse_W(self, GDD, TOD=0, FOD=0.0, w0_THz=None):
        r""" Alter the phase of the pulse 
        
        Apply the dispersion coefficients :math:`\beta_2, \beta_3, \beta_4`
        expanded around frequency :math:`\omega_0`.
        
        Parameters
        ----------
        GDD : float
             Group delay dispersion (:math:`\beta_2`) [ps^2]
        TOD : float, optional
             Group delay dispersion (:math:`\beta_3`) [ps^3], defaults to 0.
        FOD : float, optional
             Group delay dispersion (:math:`\beta_4`) [ps^4], defaults to 0.             
        w0_THz : float, optional
             Center frequency of dispersion expansion, defaults to grid center frequency.
        
        Notes
        -----
        The convention used for dispersion is
        
        .. math:: E_{new} (\omega) = \exp\left(i \left(
                                        \frac{1}{2} GDD\, \omega^2 +
                                        \frac{1}{6}\, TOD \omega^3 +
                                        \frac{1}{24} FOD\, \omega^4 \right)\right)
                                        E(\omega)
                                            
        """                

        if w0_THz is None:
            self.aw = np.exp(1j * (GDD / 2.0) * self.v_THz**2 + 
                                   1j * (TOD / 6.0) * self.v_THz**3+ 
                                   1j * (FOD / 24.0) * self.v_THz**4) * self.aw
        else:
            V = self.w_THz - w0_THz
            self.aw = np.exp(1j * (GDD / 2.0) * V**2 + 
                                   1j * (TOD / 6.0) * V**3+ 
                                   1j * (FOD / 24.0) * V**4) * self.AW 
                                   
    def apply_phase_W(self, phase):
        self.aw = self.aw * np.exp(1j*phase)
        
    def chirp_pulse_T(self, chirp2, chirp3, T0):
        self.at = self.at * np.exp(-1j * (chirp2 / 2.0) * (self.t_ps/T0)**2 + 
                                 -1j * (chirp3 / 3.0) * (self.t_ps/T0)**3) 
                                 
    # def dechirp_pulse(self, GDD_TOD_ratio = 0.0, intensity_threshold = 0.05):
    #     spect_w = self.AW
    #     phase   = np.unwrap(np.angle(spect_w))
    #     ampl    = np.abs(spect_w)
    #     mask = ampl**2 > intensity_threshold * np.max(ampl)**2
    #     gdd     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 2))
    #     self.AW( ampl * np.exp(1j*(phase-gdd(self.W_THz))) )
    #
    # def remove_time_delay(self, intensity_threshold = 0.05):
    #
    #     spect_w = self.AW
    #     phase   = np.unwrap(np.angle(spect_w))
    #     ampl    = np.abs(spect_w)
    #     mask = ampl**2 > (intensity_threshold * np.max(ampl)**2)
    #     ld     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 1))
    #     self.set_AW( ampl * np.exp(1j*(phase-ld(self.W_THz))) )
    
    # def add_time_offset(self, offset_ps):
    #     """Shift field in time domain by offset_ps picoseconds. A positive offset
    #        moves the pulse forward in time. """
    #     phase_ramp = np.exp(-1j*self.W_THz*offset_ps)
    #     self.set_AW(self.AW * phase_ramp)
    #
    # def expand_time_window(self, factor_log2, new_pts_loc = "before"):
    #     r""" Expand the time window by zero padding.
    #     Parameters
    #     ----------
    #     factor_log2 : integer
    #         Factor by which to expand the time window (1 = 2x, 2 = 4x, etc.)
    #     new_pts_loc : string
    #         Where to put the new points. Valid options are "before", "even",
    #         "after
    #     """
    #     num_new_pts = self.NPTS*(2**factor_log2 - 1)
    #     AT_current = self.AT
    #
    #     self.set_NPTS(self.NPTS * 2**factor_log2)
    #     self.set_time_window_ps(self.time_window_ps * 2**factor_log2)
    #     self._AW = None # Force generation of new array
    #     if new_pts_loc == "before":
    #         self.set_AT(np.hstack( (np.zeros(num_new_pts,), AT_current) ))
    #     elif new_pts_loc == "after":
    #         self.set_AT(np.hstack( (AT_current, np.zeros(num_new_pts,)) ))
    #     elif new_pts_loc == "even":
    #         pts_before = int(np.floor(num_new_pts * 0.5))
    #         pts_after  = num_new_pts - pts_before
    #         self.set_AT(np.hstack( (np.zeros(pts_before,),
    #                                 AT_current,
    #                                 np.zeros(pts_after,)) ))
    #     else:
    #         raise ValueError("new_pts_loc must be one of 'before', 'after', 'even'")
    #
    # def rotate_spectrum_to_new_center_wl(self, new_center_wl_nm):
    #     """Change center wavelength of pulse by rotating the electric field in
    #         the frequency domain. Designed for creating multiple pulses with same
    #         gridding but of different colors. Rotations is by integer and to
    #         the closest omega."""
    #     new_center_THz = c_nmps/new_center_wl_nm
    #     rotation = (self.center_frequency_THz-new_center_THz)/self.dF_THz
    #     self.set_AW(np.roll(self.AW, -1*int(round(rotation))))
    #
    # def interpolate_to_new_center_wl(self, new_wavelength_nm):
    #     r""" Change grids by interpolating the electric field onto a new
    #     frequency grid, defined by the new center wavelength and the current
    #     pulse parameters. This is useful when grid overlaps must be avoided,
    #     for example in difference or sum frequency generation.
    #
    #     Parameters
    #     ----------
    #     new_wavelength_nm : float
    #          New center wavelength [nm]
    #     Returns
    #     -------
    #     Pulse instance
    #     """
    #     working_pulse = self.create_cloned_pulse()
    #     working_pulse.set_center_wavelength_nm(new_wavelength_nm)
    #     interpolator = interp1d(self.W_mks, self. AW,
    #                             bounds_error = False,
    #                             fill_value = 0.0)
    #     working_pulse.set_AW(interpolator(working_pulse.W_mks))
    #     return working_pulse
    #
    # def filter_by_wavelength_nm(self, lower_wl_nm, upper_wl_nm):
    #     AW_new = self.AW
    #     AW_new[self.wl_nm < lower_wl_nm] = 0.0
    #     AW_new[self.wl_nm > upper_wl_nm] = 0.0
    #     self.set_AW(AW_new)
    
    # def clone_pulse(self, p):
    #     '''Copy all parameters of pulse_instance into this one'''
    #     self.set_NPTS(p.NPTS)
    #     self.set_time_window_ps(p.time_window_ps)
    #     self.set_center_wavelength_nm(p.center_wavelength_nm)
    #     self._frep_MHz = p.frep_MHz
    #     self.set_AT(p.AT)
    #
    # def create_cloned_pulse(self):
    #     '''Create and return new pulse instance identical to this instance.'''
    #     p = Pulse()
    #     p.clone_pulse(self)
    #     return p           
    
        #
    # def calculate_weighted_avg_frequency_mks(self):
    #     avg = np.sum(abs(self.AW)**2 * self.W_mks)
    #     weights = np.sum(abs(self.AW)**2)
    #     result = avg / (weights * 2.0 * np.pi)
    #     return result
    #
    # def calculate_weighted_avg_wavelength_nm(self):
    #     return 1.0e9 * c_mks / self.calculate_weighted_avg_frequency_mks()
    #
    # def calculate_intensity_autocorrelation(self):
    #     r""" Calculates and returns the intensity autocorrelation,
    #     :math:`\int P(t)P(t+\tau) dt`
    #
    #     Returns
    #     -------
    #     x : ndarray, shape N_pts
    #         Intensity autocorrelation. The grid is the same as the pulse class'
    #         time grid.
    #
    #     """
    #     return np.correlate(abs(self.AT)**2, abs(self.AT), mode='same')
    #
    # def spectrogram(self, gate_type='xfrog', gate_function_width_ps=0.020, time_steps=500):
    #     """This calculates a spectrogram, essentially the spectrally-resolved cross-correlation of the pulse.
    #
    #     Generally, the gate_type should set to 'xfrog', which performs a cross-correlation similar to the XFROG
    #     experiment, where the pulse is probed by a short, reference pulse. The temporal width of this pulse
    #     is set by the "gate_function_width_ps" parameter.
    #
    #     See Dudley Fig. 10, on p1153 for a description
    #     of the spectrogram in the context of supercontinuum generaiton.
    #     (http://dx.doi.org/10.1103/RevModPhys.78.1135)
    #
    #     Alternatively, the gate_type can be set to 'frog', which simulates a SHG-FROG measurement,
    #     where the pulse is probed with a copy of itself, in an autocorrelation fashion.
    #     Interpreting this FROG spectrogram is less intuitive, so this is mainly useful for comparison
    #     with experimentally recorded FROG spectra (which are often easier to acquire than XFROG measurements.)
    #
    #     A nice discussion of various FROG "species" is available here: http://frog.gatech.edu/tutorial.html
    #
    #     Parameters
    #     ----------
    #     gate_type : string
    #         Determines the type of gate function. Can be either 'xfrog' or 'frog'.
    #         Should likely be set to 'xfrog' unless comparing with experiments.
    #         See discussion above. Default is 'xfrog'.
    #     gate_function_width : float
    #         the width of the gate function in seconds. Only applies when gate_type='xfrog'.
    #         A shorter duration provides better temporal resolution, but worse spectral resolution,
    #         so this is a trade-off. Typically, 0.01 to 0.1 ps works well.
    #     time_steps : int
    #         the number of delay time steps to use. More steps makes a higher
    #         resolution spectrogram, but takes longer to process and plot.
    #         Default is 500
    #
    #     Returns
    #     -------
    #     DELAYS : 2D numpy meshgrid
    #         the columns have increasing delay (in ps)
    #     FREQS : 2D numpy meshgrid
    #         the rows have increasing frequency (in THz)
    #     spectrogram : 2D numpy array
    #         Following the convention of Dudley, the frequency runs along the y-axis
    #         (axis 0) and the time runs alon the x-axis (axis 1)
    #
    #     Example
    #     -------
    #     The spectrogram can be visualized using something like this: ::
    #
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         DELAYS, FREQS, extent, spectrogram = pulse.spectrogram()
    #         plt.imshow(spectrogram, aspect='auto', extent=extent)
    #         plt.xlabel('Time (ps)')
    #         plt.ylabel('Frequency (THz)')
    #         plt.tight_layout
    #
    #         plt.show()
    #
    #     output:
    #
    #     .. image:: https://cloud.githubusercontent.com/assets/1107796/13677657/25075ea4-e6a8-11e5-98b4-7813fa9a6425.png
    #        :width: 500px
    #        :alt: example_result
    #     """
    #
    #     def gauss(x, A=1, mu=0, sigma=1): # gaussian function
    #         return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    #
    #     t = self.T_ps # working in ps
    #
    #     delay = np.linspace(np.min(t), np.max(t), time_steps)
    #     D, T  = np.meshgrid(delay, t)
    #     D, AT = np.meshgrid(delay, self.AT)
    #
    #     phase = np.unwrap(np.angle(AT))
    #     amp   = np.abs(AT)
    #
    #
    #     if gate_type == 'xfrog':
    #         gate_function = gauss(T, mu=D, sigma=gate_function_width_ps)
    #     elif gate_type=='frog':
    #         dstep = float(delay[1]-delay[0])
    #         tstep = float(    t[1]-    t[0])
    #         # calculate the coordinates of the new array
    #         dcoord = D*0
    #         tcoord = (T-D-np.min(T))/tstep
    #
    #         # gate_function = scipy.ndimage.interpolation.map_coordinates(amp, (tcoord, dcoord))
    #
    #         gate_function_real = scipy.ndimage.interpolation.map_coordinates(np.real(AT), (tcoord, dcoord))
    #         gate_function_imag = scipy.ndimage.interpolation.map_coordinates(np.imag(AT), (tcoord, dcoord))
    #         gate_function = gate_function_real + 1j*gate_function_imag
    #
    #     else:
    #         raise ValueError('Type \""%s\"" not recognized. Type must be \"xfrog\" or \"frog\".'%gate_type)
    #
    #     # make a 2D array of E(time, delay)
    #     E = amp * gate_function * np.exp(1j*(2 * np.pi * T * self.center_frequency_THz + phase) )
    #
    #     spectrogram = np.fft.fft(E, axis=0)
    #     freqs = np.fft.fftfreq(np.shape(E)[0], t[1]-t[0])
    #
    #     DELAYS, FREQS = np.meshgrid(delay, freqs)
    #
    #     # just take positive frequencies:
    #     h = np.shape(spectrogram)[0]
    #     spectrogram = spectrogram[:h//2]
    #     DELAYS      = DELAYS[:h//2]
    #     FREQS       = FREQS[:h//2]
    #
    #     # calculate the extent to make it easy to plot:
    #     extent = (np.min(DELAYS), np.max(DELAYS), np.min(FREQS), np.max(FREQS))
    #
    #     return DELAYS, FREQS, extent, np.abs(spectrogram)
    #
    #
