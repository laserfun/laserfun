import laserfun as lf
import numpy as np
import os
import sys


# speed of light in m/s and nm/ps
c_mks = 299792458.0
c_nmps = c_mks * 1e9/1e12

def test_pulse():
    """Test that the max of pulses match previously calculated values."""

    parameters = dict(center_wavelength_nm=1550, time_window_ps=20,
                      fwhm_ps=0.1, GDD=0, TOD=0, npts=2**13,
                      frep_MHz=100, power_is_avg=False, epp=50e-12)

    # test if the pulse FWHM is correctly implemented:
    pulse = lf.Pulse(pulse_type='gaussian', **parameters)
    assert np.isclose(pulse.calc_width(level=0.5), 0.1, rtol=1e-3)

    pulse = lf.Pulse(pulse_type='sech', **parameters)
    assert np.isclose(pulse.calc_width(level=0.5), 0.1, rtol=2e-3)

    pulse = lf.Pulse(pulse_type='sinc', **parameters)
    assert np.isclose(pulse.calc_width(level=0.5), 0.1, rtol=1e-3)

    # try to make a pulse with power_is_avg
    pulse = lf.Pulse(pulse_type='sech', power_is_avg=True, time_window_ps=20)

    assert np.abs(pulse.time_window_ps - 20) < 1e-5

    # test if the add-noise functions work without crashing
    pulse.add_noise(noise_type='sqrt_N_freq')
    pulse.add_noise(noise_type='one_photon_freq')


def test_pulse_psd():
    """Test the pulse.psd function with the various power-spectral-density
    types to ensure that each integrates to the correct average power."""

    pulse = lf.Pulse(pulse_type='gaussian', center_wavelength_nm=1550,
                     fwhm_ps=0.02, time_window_ps=1, npts=2**13, frep_MHz=100,
                     epp=50e-12)

    rr = 1e8
    mW = pulse.epp * rr * 1e3  # actual average power
    df = pulse.f_THz[1] - pulse.f_THz[0]
    tol = 1e-9  # absolute tolerance
    wl = pulse.wavelength_nm

    # now, calculate the average power by integrating each PSD method

    assert np.abs(np.trapz(pulse.psd('mW/bin', rep_rate=rr)) - mW) < tol
    assert np.abs(np.trapz(pulse.psd('mW/THz', rep_rate=rr)*df) - mW) < tol

    dBm_per_THz = pulse.psd('dBm/THz', rep_rate=rr)
    assert np.trapz(10**(0.1*dBm_per_THz)) * df - mW < tol

    mW_per_nm = pulse.psd('mW/nm', rep_rate=rr)
    assert np.trapz(mW_per_nm, x=wl) - mW < tol

    dBm_per_nm = pulse.psd('dBm/nm', rep_rate=rr)
    assert np.trapz(10**(0.1*dBm_per_nm), x=wl) - mW < tol


def test_fiber():
    fiber = lf.Fiber(1, center_wl_nm=1550)

    def disp(z):
        return [1]

    def myfunc(z):
        return 1

    fiber.set_dispersion_function(disp)
    fiber.set_gamma_function(myfunc)
    fiber.set_alpha_function(myfunc)

    pulse = lf.Pulse(pulse_type='sech')

    assert np.all(np.isfinite(fiber.get_B(pulse, z=0.5)))
    assert fiber.get_gamma(z=0.5) == 1
    assert fiber.get_alpha(z=0.5) == 1


def test_nlse_spectrum():
    """Check for reasonable agreement of NLSE output with old PyNLO results."""
    pulse = lf.Pulse(pulse_type='sech', fwhm_ps=0.050, time_window_ps=7,
                     npts=2**12, center_wavelength_nm=1550.0, epp=50e-12)

    fiber1 = lf.Fiber(length=8e-3, center_wl_nm=1550.0, gamma_W_m=1,
                      dispersion=(-0.12, 0, 5e-6))

    results = lf.NLSE(pulse, fiber1, raman=True, shock=True, nsaves=100,
                      atol=1e-5, rtol=1e-5, integrator='lsoda',
                      print_status=False)

    z, f, t, AW, AT = results.get_results()
    dB = 10*np.log10(np.abs(AW[-1])**2)
    path = os.path.split(os.path.realpath(__file__))[0]
    f_prev, dB_prev = np.loadtxt(path+'/nlse_output.txt', delimiter=',',
                                 unpack=True, skiprows=1)

    # this is probably overly stringent because we are on a dB scale
    np.testing.assert_allclose(dB, dB_prev, rtol=1e-4, atol=0)

    # test that the wavelengths function works:
    z, new_wls, t, AW_wls, AT = results.get_results_wavelength()
    assert np.all(np.isfinite(AW_wls))


def test_nlse_loss():
    """Check that the loss is applied correctly in the NLSE."""
    # create a 1 meter fiber with 3.01 dB (50%) loss per meter:
    loss = 3.0102999566
    fiber = lf.Fiber(length=1, loss_dB_per_m=loss)
    pulse = lf.Pulse(epp=100e-12)  # create pulse
    results = lf.NLSE(pulse, fiber, print_status=False)

    assert np.isclose(pulse.epp * 0.5, results.pulse_out.epp, rtol=1e-11)
    
def test_nlse_psd():
    """Check all of the power-spectral density (PSD) calculations in the NLSE
    and ensure that the pulse energy or average power is maintained."""
    
    epp = 100e-12
    rr = 100e6  # rep rate
    mW = epp * rr * 1e3  # average power
    k = dict(rtol=1e-13, atol=0) # tolerances

    
    # run a simple propagation without any loss:
    fiber = lf.Fiber(length=.1, loss_dB_per_m=0)
    pulse = lf.Pulse(epp=epp, fwhm_ps=0.02)
    r = lf.NLSE(pulse, fiber, print_status=False)
    
    z, f, t, AW, AT = r.get_results(data_type='mW/bin', rep_rate=rr)
    assert np.allclose(np.trapz(AW, axis=1), mW, **k)
    assert np.allclose(np.trapz(AT, axis=1), mW, **k)
        
    z, f, t, AW, AT = r.get_results(data_type='mW/THz', rep_rate=rr)
    df = f[1] - f[0]
    dt = t[1] - t[0]
    assert np.allclose(np.trapz(AW*df, axis=1), mW, **k)
    assert np.allclose(np.trapz(AT*dt, axis=1), mW, **k)
    
    z, f, t, AW, AT = r.get_results(data_type='dBm/THz', rep_rate=rr)
    assert np.allclose(np.trapz(10**(0.1*AW)*df, axis=1), mW, **k)
    assert np.allclose(np.trapz(10**(0.1*AT)*dt, axis=1), mW, **k)

    z, f, t, AW, AT = r.get_results(data_type='mW/nm', rep_rate=rr)
    wl = c_nmps/f
    trapz = np.trapz(AW[:,::-1], axis=1, x=wl[::-1])
    assert np.allclose(trapz, mW, rtol=1e-6, atol=0)
    assert np.allclose(np.trapz(AT*dt, axis=1), mW, **k)
    
    z, f, t, AW, AT = r.get_results(data_type='dBm/nm', rep_rate=rr)
    wl = c_nmps/f
    trapz = np.trapz(10**(0.1*AW)[:,::-1], axis=1, x=wl[::-1])
    assert np.allclose(trapz, mW, rtol=1e-6, atol=0)
    assert np.allclose(np.trapz(10**(0.1*AT)*dt, axis=1), mW, **k)

    # TODO: figure out why the "per nm" tests require higher tolerances

def test_examples():
    sys.path.append(__file__+'../../examples')
    import examples
    print('Tested all examples in: ' + os.path.split(examples.__file__)[0])

if __name__ == '__main__':
    print('test_pulse...')
    test_pulse()
    print('test_pulse_psd...')
    test_pulse_psd()
    print('test_nlse_loss...')
    test_nlse_loss()
    print('test_nlse_spectrum...')
    test_nlse_spectrum()
    print('test_fiber...')
    test_fiber()
    print('test_nlse_psd')
    test_nlse_psd()
    print('Testing examples...')
    import sys
    sys.path.append(__file__+'../../examples')
    import examples
    print('Tested all examples in: ' + os.path.split(examples.__file__)[0])
    print('Tests complete!')
