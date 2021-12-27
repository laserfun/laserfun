import numpy as np
import matplotlib.pyplot as plt
import nlse

FWHM    = 0.050  # pulse duration (ps)
pulseWL = 1550   # pulse central wavelength (nm)
EPP     = 50e-12 # Energy per pulse (J)
GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0      # Third order dispersion (ps^3)

Window  = 7.0    # simulation window (ps)
Steps   = 200    # simulation steps
Points  = 2**12  # simulation points
rtol    = 1e-4   # relative error
atol    = 1e-4   # absolute error

# initial properties (at the start of the fiber)
beta2i   = -120   # (ps^2/km)
beta3i   = 0.00   # (ps^3/km)
beta4i   = 0.005  # (ps^4/km)
Gammai   = 1000   # nonlinearity (1/(W km))
Alphai   = 0      # loss (dB/cm)

# final properties (at the end of the fiber)
beta2f   = -120  # (ps^2/km)
beta3f   = 0.00   # (ps^3/km)
beta4f   = 0.005  # (ps^4/km)
Gammaf   = 2000   # nonlinearity (1/(W km))
Alphaf   = 2     # loss (dB/cm)

Length  = 10     # length in mm

fibWL   = pulseWL  # Center WL of fiber (nm)

Raman   = True    # Enable Raman effect?
Steep   = True    # Enable self steepening?

# set up plots for the results:
fig = plt.figure(figsize=(8,8))
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

# create the pulse
pulse = nlse.pulse.Pulse(pulse_type='sech', power=1, fwhm_ps=FWHM, center_wavelength_nm=pulseWL,
                             time_window_ps=Window, GDD=GDD, TOD=TOD, npts= Points,
                             frep_MHz=100, power_is_avg=False, epp=EPP)

# create the fiber!
fiber1 = nlse.Fiber(Length * 1e-3, center_wl_nm=fibWL, dispersion=(beta2i*1e-3, beta3i*1e-3, beta4i*1e-3),
                      gamma_W_m=Gammai * 1e-3, loss_dB_per_m=Alphai)

def dispersion_function(z):
    frac = z/(Length*1e-3)
    beta2 = (1-frac) * beta2i + frac * beta2f
    beta3 = (1-frac) * beta3i + frac * beta3f
    beta4 = (1-frac) * beta4i + frac * beta4f
    return (beta2*1e-3, beta3*1e-3, beta4*1e-3)

fiber1.set_dispersion_function(dispersion_function)

def gamma_function(z):
    frac = z/(Length*1e-3)
    gamma = 1e-3*((1-frac)*Gammai + frac*Gammaf)
    return gamma

fiber1.set_gamma_function(gamma_function)

def alpha_function(z):
    frac = z/(Length*1e-3)
    alpha = ((1-frac)*Alphai + frac*Alphaf)
    return alpha*100

fiber1.set_alpha_function(alpha_function)

# propagate the pulse using the NLSE
results = nlse.NLSE.nlse(pulse, fiber1,raman=Raman,
                              shock=Steep, nsaves=Steps,
                              atol=atol, rtol=rtol, integrator='lsoda', reload_fiber=True)

z, f, t, AT, AW = results.get_results() # unpack results

z = z * 1e3  # convert to mm

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

IW_dB = dB(AW)
IT_dB = dB(AT)

ax0.plot(f, dB(pulse.aw), color = 'b', label='Initial pulse')
ax1.plot(t, dB(pulse.at), color = 'b', label='Initial pulse')

ax0.plot(f, IW_dB[-1], color='r', label='Final pulse')
ax1.plot(t, IT_dB[-1], color='r', label='Final pulse')

ax1.legend(loc='upper left', fontsize=9)


ax0.set_xlabel('Frequency (THz)')
ax1.set_xlabel('Time (ps)')

ax0.set_ylabel('Intensity (dB)')
ax0.set_ylim( -140,  0)
ax1.set_ylim( -70, 40)
ax1.set_xlim(-1.5, 1.5)

ax2.set_ylabel('Propagation distance (mm)')
ax2.set_xlabel('Frequency (THz)')
ax2.set_xlim(0, 400)

extf = (np.min(f), np.max(f), np.min(z), np.max(z))
extt = (np.min(t), np.max(t), np.min(z), np.max(z))

ax2.imshow(IW_dB, extent=extf, vmin=np.max(IW_dB) - 40.0, vmax=np.max(IW_dB), aspect='auto', origin='lower')
ax3.imshow(IT_dB, extent=extt, vmin=np.max(IT_dB) - 40.0, vmax=np.max(IT_dB), aspect='auto', origin='lower')

ax3.set_xlabel('Frequency (THz)')

fig.tight_layout()

plt.show()
