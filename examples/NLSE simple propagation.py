import laserfun as lf

pulse = lf.Pulse(pulse_type='sech', fwhm_ps=0.050, epp=50e-12, center_wavelength_nm=1550)
fiber1 = lf.Fiber(length=0.010, center_wl_nm=1550, dispersion=(-0.12, 0, 5e-6), gamma_W_m=1)
results = lf.NLSE(pulse, fiber1)

results.plot()