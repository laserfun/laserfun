import numpy as np
import matplotlib.pyplot as plt
import nlse
import time
import pynlo

FWHM    = 0.050  # pulse duration (ps)
pulseWL = 1550   # pulse central wavelength (nm)
EPP     = 50e-12 # Energy per pulse (J)
GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0  # Third order dispersion (ps^3)

Window  = 7.0   # simulation window (ps)
Steps   = 100     # simulation steps
Points  = 2**12  # simulation points
error   = 0.001

beta2   = -120     # (ps^2/km)
beta3   = 0.00     # (ps^3/km)
beta4   = 0.005    # (ps^4/km)
        
Length  = 15    # length in mm
    
Alpha   = 0    # attentuation coefficient (dB/cm)
Gamma   = 1000    # Gamma (1/(W km) 
    
fibWL   = pulseWL # Center WL of fiber (nm)
    
Raman   = True   # Enable Raman effect?
Steep   = True    # Enable self steepening?

alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m

# set up plots for the results:
fig = plt.figure(figsize=(8,8))
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2)

# create the pulse
pulse = nlse.pulse.SechPulse(power=1, T0_ps=FWHM/1.76, center_wavelength_nm=pulseWL, 
                             time_window_ps=Window, GDD=GDD, TOD=TOD, NPTS= Points, 
                             frep_MHz=100, power_is_avg=False)

pulse.set_epp(EPP)  # set the pulse energy

t = pulse.T_ps

# create the fiber!
fiber1 = nlse.fiber.FiberInstance()
fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                      gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)

# run the new method:
t_start = time.time()
results = nlse.NLSE.nlse(pulse, fiber1, loss=alpha, raman=Raman, 
                              shock=Steep, flength=Length*1e-3, nsaves=Steps,
                              atol=1e-5, rtol=1e-5, integrator='lsoda', reload_fiber=False)
z, AT, AW, w = results.get_results()


t_nlse = time.time() - t_start

z = z * 1e3  # convert to mm 
f = w
IW_dB = 10*np.log10(np.abs(AW)**2)
IT_dB = 10*np.log10(np.abs(AT)**2)

# run the PyNLO method
t_start = time.time()
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=error, USE_SIMPLE_RAMAN=True,
                 disable_Raman = np.logical_not(Raman),
                 disable_self_steepening = np.logical_not(Steep))
y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)
t_ssfm = time.time() - t_start

# Now we make some plots:

F = pulse.F_THz  # Frequency grid of pulse (THz)

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

zW = dB( np.transpose(AW)[:, (F > 0)] )
zT = dB( np.transpose(AT) )

y_mm = y * 1e3 # convert distance to mm

ax0.plot(pulse.F_THz, dB(pulse.AW),  color = 'b', label='Initial pulse')
ax1.plot(pulse.T_ps,  dB(pulse.AT),  color = 'b', label='Initial pulse')

ax0.plot(pulse_out.F_THz, dB(pulse_out.AW),  color = 'r', label='PyNLO SSFM')
ax1.plot(pulse_out.T_ps,  dB(pulse_out.AT),  color = 'r', label='PyNLO SSFM')

extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, Length)
ax2.imshow(zW, extent=extent,
           vmin=np.max(zW) - 40.0, vmax=np.max(zW),
           aspect='auto', origin='lower')

ax2.set_title('PyNLO SSFM: %.2f sec'%t_ssfm)
ax3.set_title('Dudley: %.2f sec'%t_nlse)

ax0.set_ylabel('Intensity (dB)')
ax0.set_ylim( - 140,  0)
ax1.set_ylim( - 70, 40)

ax2.set_ylabel('Propagation distance (mm)')
ax2.set_xlabel('Frequency (THz)')
ax2.set_xlim(0, 400)

# plot new  method:

ax0.plot(f, IW_dB[-1], color='C1', label='Dudley')
ax1.plot(t, IT_dB[-1], color='C1', label='Dudley')

ax0.set_xlabel('Frequency (THz)')
ax1.set_xlabel('Time (ps)')

extent = (np.min(f), np.max(f), np.min(z), np.max(z))
ax3.imshow(IW_dB, extent=extent,
           vmin=np.max(IW_dB) - 40.0, vmax=np.max(IW_dB),
           aspect='auto', origin='lower')

ax3.set_xlabel('Frequency (THz)')
ax3.set_xlim(0, 400)

ax1.legend(loc='upper left', fontsize=9)

fig.tight_layout()

wls, AW_WL = results.get_amplitude_wavelengths()
AW_WL = np.sqrt(AW_WL)

extent = (np.min(wls), np.max(wls), np.min(z), np.max(z))

plt.figure()
plt.imshow(dB(AW_WL), extent=extent, origin='lower', aspect='auto', vmin=np.max(dB(AW_WL)) - 40.0, vmax=np.max(dB(AW_WL)))



plt.show()