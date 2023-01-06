"""Runs a simple example of NLSE pulse propagation."""

import laserfun as lf

p = lf.Pulse(pulse_type='sech', fwhm_ps=0.05, epp=50e-12,
             center_wavelength_nm=1550, time_window_ps=7)

f = lf.Fiber(length=0.010, center_wl_nm=1550, dispersion=(-0.12, 0, 5e-6),
             gamma_W_m=1)

results = lf.NLSE(p, f, print_status=False)

if __name__ == '__main__':  # make plots if we're not running tests
    results.plot(units='dBm/nm')
