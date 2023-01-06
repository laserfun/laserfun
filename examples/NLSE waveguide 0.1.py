"""Demonstrates how to load the dispersion from a separate file. In this case,
   the disperison corresponds to a silicon nitride waveguide."""

import numpy as np
import laserfun as lf
import scipy.interpolate

# waveguide parameters
disp_file = 'Si3N4_oxideclad_thick-750nm_widths-0.4-3.9_wav-0.40-3.00um.npz'
width = 2.4  # micron
n2 = 2.0e-19  # m^2/W
Length = 10   # length in mm
Alpha = 0     # loss (dB/cm)

# pulse parameters
FWHM = 0.100    # pulse duration (ps)
pulseWL = 1550  # pulse central wavelength (nm)
EPP = 100e-12   # Energy per pulse (J)
GDD = 0.0       # Group delay dispersion (ps^2)
TOD = 0.0       # Third order dispersion (ps^3)

# simulation parameters
Window = 5.0    # simulation window (ps)
Steps = 100     # simulation steps
Points = 2**12  # simulation points
rtol = 1e-4     # relative error for NLSE integrator
atol = 1e-4     # absolute error
Raman = False   # Enable Raman effect?
Steep = True    # Enable self steepening?

# load the dispersion file:
data = np.load(disp_file)
wls = data['wav']*1e3
aeff_int = scipy.interpolate.interp1d(data['widths'], data['aeff'], axis=0)
neff_int = scipy.interpolate.interp1d(data['widths'], data['neff'], axis=0)
aeff = aeff_int(width)
neff = neff_int(width)


def disp_function(z=0):  # provide effective index to the NLSE
    return (wls, neff)


def gamma_function(z=0):  # provide the nonlinearity at the pump to the NLSE
    aeff_interp = scipy.interpolate.interp1d(wls, aeff)
    return 2*np.pi*n2/(pulseWL*1e-9*aeff_interp(pulseWL)*1e-12)


# create the pulse:
p = lf.Pulse(pulse_type='sech', fwhm_ps=FWHM, center_wavelength_nm=pulseWL,
             time_window_ps=Window, GDD=GDD, TOD=TOD, npts=Points, epp=EPP)

# create the fiber
f = lf.Fiber(Length * 1e-3, center_wl_nm=pulseWL, dispersion_format='GVD',
             gamma_W_m=gamma_function(), loss_dB_per_m=Alpha*100)

f.set_dispersion_function(disp_function, dispersion_format='n')

# propagate the pulse using the NLSE
results = lf.NLSE(p, f, raman=Raman, shock=Steep, nsaves=Steps, rtol=rtol,
                  atol=atol, print_status=True)

results.plot(wavelength=True, units='dBm/nm', show=True, tlim=(-2, 2))
