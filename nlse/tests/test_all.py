import nlse
import numpy as np
import os
import matplotlib.pyplot as plt

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
    fiber = nlse.Fiber(1e-3, center_wl_nm=1550)
    
    def myFunction(z):
        return 1
    fiber.set_dispersion_function(myFunction)
    fiber.set_gamma_function(myFunction)
    fiber.set_alpha_function(myFunction)
    # fiber.get_B()
    
    
def test_nlse_spectrum():
    """
    This compares the NLSE output with previous results that were benchmarked
    agaist PyNLO and found to be in good agreement in the regions with amplitude."""    
    beta2   = -120 * 1e-3 # (ps^2/m)
    beta3   = 0.00 * 1e-3 # (ps^3/m)
    beta4   = 0.005* 1e-3 # (ps^4/km)
    
    pulse = nlse.pulse.Pulse(pulse_type='sech', fwhm_ps=0.050, time_window_ps=7, 
                             npts=2**12, center_wavelength_nm=1550.0, epp=50e-12)

    fiber1 = nlse.Fiber(length=8e-3, center_wl_nm=1550.0, gamma_W_m=1,
                        dispersion=(beta2, beta3, beta4))
                         
    results = nlse.NLSE.nlse(pulse, fiber1, raman=True, shock=True, nsaves=100,
                             atol=1e-5, rtol=1e-5, integrator='lsoda',
                             print_status=False)
    
    z, AT, AW, w = results.get_results()
    dB = 10*np.log10(np.abs(AW[-1])**2)
    path = os.path.split(os.path.realpath(__file__))[0]
    f_prev, dB_prev = np.loadtxt(path+'/nlse_output.txt', delimiter=',', unpack=True, skiprows=1)
    
    # this is probably overly stringent because we are on a dB scale
    np.testing.assert_allclose(dB, dB_prev, rtol=1e-4, atol=0)
    
    results.get_amplitude_wavelengths()

def test_nlse_loss():
    """
    This checks that the loss is applied correctly in the NLSE. 
    """    
    # create a 1 meter fiber with 3.01 dB (50%) loss per meter:
    loss = 3.0102999566
    fiber = nlse.Fiber(length=1, loss_dB_per_m=loss)

    pulse = nlse.pulse.Pulse(epp=100e-12)  # create pulse
    
    results = nlse.NLSE.nlse(pulse, fiber, print_status=False)
    
    assert np.abs(pulse.epp * 0.5 - results.pulse_out.epp)/pulse.epp < 1e-11
    

    
if __name__ == '__main__':
    test_pulse()
    test_nlse_loss()
    test_nlse_spectrum()
    test_fiber()