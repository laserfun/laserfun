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

def test_dispersion_tools():
    """Test dispersion conversion tools."""
    from laserfun import tools
    
    # Test values (approximate)
    wl = 1550 # nm
    D = 5 # ps/nm/km [SMF-28]
    S = 0.025 # ps/nm^2/km
    
    # Calculate beta2
    beta2 = tools.D2_to_beta2(wl, D)
    
    # Calculate D back
    D_calc = tools.beta2_to_D2(wl, beta2)
    assert np.isclose(D, D_calc), "D -> beta2 -> D failed"
    
    # Calculate beta3
    beta3 = tools.D3_to_beta3(wl, D, S)
    
    # Calculate S back
    S_calc = tools.beta3_to_D3(wl, beta3, D2=D)
    assert np.isclose(S, S_calc), "S -> beta3 -> S failed"

    # Explicit check for beta2 and beta3 values
    expected_beta2 = -6.377240954930596
    expected_beta3 = 0.051164480183629686
    
    assert np.isclose(beta2, expected_beta2), f"Expected beta2={expected_beta2}, got {beta2}"
    assert np.isclose(beta3, expected_beta3), f"Expected beta3={expected_beta3}, got {beta3}"


def test_compressor_pynlo():
    """Test TreacyCompressor against values from PyNLO."""
    from laserfun import tools
    
    wl = 1545.0 # nm
    lines_per_mm = 1000
    incident_angle_deg = 50.5789
    separations = np.array([0.0, 0.1, 0.2, 0.3])
    
    # Expected results (from PyNLO, converted to ps^n)
    expected_gdd = np.array([0.00000000e+00, -5.10085847e-24, -1.02017169e-23, -1.53025754e-23]) * 1e24
    expected_tod = np.array([0.00000000e+00, 4.97008481e-38, 9.94016963e-38, 1.49102544e-37]) * 1e36
    
    compressor = tools.TreacyCompressor(lines_per_mm=lines_per_mm, incident_angle_degrees=incident_angle_deg)
    
    calc_gdd = []
    calc_tod = []
    
    for sep in separations:
        # order 2 = GDD (now returns ps^2)
        gdd = compressor.calc_dispersion(wl, sep, 2)
        calc_gdd.append(gdd)
        
        # order 3 = TOD (now returns ps^3)
        tod = compressor.calc_dispersion(wl, sep, 3)
        calc_tod.append(tod)
        
    calc_gdd = np.array(calc_gdd)
    calc_tod = np.array(calc_tod)
    
    # Verify GDD
    # rtol=1e-4 should be sufficient given floating point diffs
    assert np.allclose(calc_gdd, expected_gdd, rtol=1e-4), \
        f"GDD mismatch.\nExpected: {expected_gdd}\nGot: {calc_gdd}"
        
    # Verify TOD
    assert np.allclose(calc_tod, expected_tod, rtol=1e-4), \
        f"TOD mismatch.\nExpected: {expected_tod}\nGot: {calc_tod}"


def test_compressor_littrow():
    """Test TreacyCompressor Littrow initialization."""
    from laserfun import tools
    
    wl_nm = 1030
    lines = 1000
    
    # Init at Littrow
    comp = tools.TreacyCompressor(lines_per_mm=lines, littrow_wavelength_nm=wl_nm)
    
    # Calculate expected Littrow angle
    # sin(theta) = lambda / 2d
    d = 1e-3 / lines
    expected_theta_rad = np.arcsin(wl_nm * 1e-9 / (2 * d))
    
    assert np.isclose(comp.g, expected_theta_rad), f"Littrow angle mismatch. Got {comp.g}, expected {expected_theta_rad}"
    
    # Verify we can calc GDD without error
    gdd = comp.calc_dispersion(wl_nm, 0.1, order=2)
    assert np.isfinite(gdd)


def test_compressor_phase_consistency():
    """
    Check that the phase applied by the TreacyCompressor (full grating phase)
    is consistent with the GDD, TOD, and FOD calculated from the same parameters.
    """
    from laserfun import tools

    # Parameters
    wl = 1545.0  # nm
    lines = 1000  # l/mm
    separation = 4e-3  # 4 mm
    fwhm_ps = 0.1  # 100 fs

    # Initialize Compressor (Littrow)
    comp = tools.TreacyCompressor(lines_per_mm=lines, littrow_wavelength_nm=wl)

    # Setup Pulses
    npts = 2**14
    time_window = 40.0  # ps

    # Pulse 1: Apply full compressor phase
    p1 = lf.Pulse(pulse_type='gaussian', fwhm_ps=fwhm_ps, center_wavelength_nm=wl,
                  time_window_ps=time_window, npts=npts)
    comp.apply_phase_to_pulse(separation, p1)

    # Calculate GDD, TOD, and FOD for approximation
    gdd, tod, fod = comp.calc_dispersion(wl, separation, order=[2, 3, 4])

    # Pulse 2: Create with GDD, TOD, and FOD directly
    p2 = lf.Pulse(pulse_type='gaussian', fwhm_ps=fwhm_ps, center_wavelength_nm=wl,
                  time_window_ps=time_window, npts=npts,
                  GDD=gdd, TOD=tod, FOD=fod)

    # Compare full intensity profiles
    # 1. Mean difference (sensitivity to overall phase curvature)
    # Benchmarked mean relative diff is ~1.2e-4, so 1.0e-3 is a tight, safe tolerance.
    mean_diff = np.mean(np.abs(p1.it - p2.it)) / np.max(p1.it)
    assert mean_diff < 1.0e-3, f"Pulse profile mean mismatch: {mean_diff:.6e}"

    # 2. Point-by-point difference (ensures no local glitches)
    # Benchmarked max relative diff is ~6.4e-4, so 1.0e-2 is a robust point-by-point tolerance.
    max_diff = np.max(np.abs(p1.it - p2.it)) / np.max(p1.it)
    assert max_diff < 1.0e-2, f"Pulse profile point-by-point mismatch: {max_diff:.6e}"


def test_examples():
    """Test that all examples can be imported and run without error."""
    # Add project root to path so we can import 'examples'
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if root not in sys.path:
        sys.path.insert(0, root)
    
    import examples
    if hasattr(examples, '__file__') and examples.__file__:
        print('Tested all examples in: ' + os.path.dirname(examples.__file__))

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
    print('test_dispersion_tools...')
    test_dispersion_tools()
    print('test_compressor_pynlo...')
    test_compressor_pynlo()
    print('test_compressor_littrow...')
    test_compressor_littrow()
    print('test_compressor_phase_consistency...')
    test_compressor_phase_consistency()
    
    print('Testing examples...')
    test_examples()

    print('Tests complete!')
