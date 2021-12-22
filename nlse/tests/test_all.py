import nlse
import numpy as np
import os

def test_pulse():
    """Test that the maxima of a gaussian and sech pulse match their 
    previously calculated values."""
    parameters = dict(center_wavelength_nm=1550, time_window_ps=20, 
                      fwhm_ps=0.5, GDD=0, TOD=0, npts=2**13,
                      frep_MHz=100, power_is_avg=False, epp=50e-12)

    pulse = nlse.pulse.Pulse(pulse_type='gaussian', **parameters)
    aw = np.abs(pulse.aw)**2
    assert np.abs(np.max(aw) - 0.075)/0.075 < 0.01

    pulse = nlse.pulse.Pulse(pulse_type='sech', **parameters)
    aw = np.abs(pulse.aw)**2
    assert np.abs(np.max(aw) - 0.175)/0.175 < 0.01

    pulse = nlse.pulse.Pulse(pulse_type='sinc', **parameters)
    aw = np.abs(pulse.aw)**2
    assert np.abs(np.max(aw) - 0.0608)/0.0608 < 0.01
    
    # try to make a pulse with power_is_avg
    pulse = nlse.pulse.Pulse(pulse_type='sech', power_is_avg=True, time_window_ps=20)
    
    assert np.abs(pulse.time_window_ps - 20) < 1e-5
    
    pulse.add_noise(noise_type='sqrt_N_freq')
    pulse.add_noise(noise_type='one_photon_freq')

def test_fiber():
    fiber = nlse.fiber.FiberInstance()
    fiber.generate_fiber(1e-3, center_wl_nm=1550, betas=(0, 0, 0))
    
    def myFunction(z):
        return 1
    fiber.set_dispersion_function(myFunction)
    fiber.set_gamma_function(myFunction)
    fiber.set_alpha_function(myFunction)
    # fiber.get_B()
    
    
    
def test_nlse():
    """
    This compares the NLSE output with previous results that were benchmarked
    agaist PyNLO and found to be in good agreement in the regions with amplitude."""    
    FWHM    = 0.050  # pulse duration (ps)
    pulseWL = 1550   # pulse central wavelength (nm)
    EPP     = 50e-12 # Energy per pulse (J)
    GDD     = 0.0    # Group delay dispersion (ps^2)
    TOD     = 0      # Third order dispersion (ps^3)

    Window  = 7.0    # simulation window (ps)
    Steps   = 100    # simulation steps
    Points  = 2**12  # simulation points
    rtol    = 1e-4   # relative error
    atol    = 1e-4   # absolute error

    beta2   = -120   # (ps^2/km)
    beta3   = 0.00   # (ps^3/km)
    beta4   = 0.005  # (ps^4/km)

    Length  = 8     # length in mm

    Alpha   = 0      # loss (dB/cm)
    Gamma   = 1000   # nonlinearity (1/(W km))

    fibWL   = pulseWL  # Center WL of fiber (nm)

    Raman   = True    # Enable Raman effect?
    Steep   = True    # Enable self steepening?

    alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    
    # create the pulse
    pulse = nlse.pulse.Pulse(pulse_type='sech', power=1, fwhm_ps=FWHM, center_wavelength_nm=pulseWL,
                                 time_window_ps=Window, GDD=GDD, TOD=TOD, npts= Points,
                                 frep_MHz=100, power_is_avg=False, epp=EPP)

    # create the fiber!
    fiber1 = nlse.fiber.FiberInstance()
    fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                          gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)

    # run the new method:
    results = nlse.NLSE.nlse(pulse, fiber1, loss=alpha, raman=Raman,
                                  shock=Steep, flength=Length*1e-3, nsaves=Steps,
                                  atol=1e-5, rtol=1e-5, integrator='lsoda', reload_fiber=False)
    
    z, AT, AW, w = results.get_results()
    dB = 10*np.log10(np.abs(AW[-1])**2)
    path = os.path.split(os.path.realpath(__file__))[0]
    f_prev, dB_prev = np.loadtxt(path+'/nlse_output.txt', delimiter=',', unpack=True, skiprows=1)
    
    # this is probably overly stringent because we are on a dB scale
    np.testing.assert_allclose(dB, dB_prev, rtol=1e-4, atol=0)
    
    results.get_amplitude_wavelengths()
    
    
if __name__ == '__main__':
    test_pulse()
    test_nlse()
    test_fiber()