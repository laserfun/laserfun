"""Runs a simple example of NLSE pulse propagation and spectrogram plotting."""

import laserfun as lf
import numpy as np

p = lf.Pulse(pulse_type='sech', fwhm_ps=0.05, epp=50e-12,
             center_wavelength_nm=1550, time_window_ps=7)

f = lf.Fiber(length=0.010, center_wl_nm=1550, dispersion=(-0.12, 0, 5e-6),
             gamma_W_m=1)

results = lf.NLSE(p, f, print_status=False)

pulse = results.pulse_out

if __name__ == '__main__':  # make plots if we're not running tests
    results.plot(units='dBm/nm')
    pulse.plot_spectrogram(wavelength_or_frequency='frequency')
    
    results.plot(units='dBm/nm',wavelength=True)
    pulse.plot_spectrogram(wavelength_or_frequency='wavelength')