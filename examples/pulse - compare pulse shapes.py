import matplotlib.pyplot as plt
import numpy as np
import nlse

FWHM    = 0.50  # pulse duration (ps)
pulseWL = 1550   # pulse central wavelength (nm)
EPP     = 50e-12 # Energy per pulse (J)
GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0      # Third order dispersion (ps^3)
Window  = 20.0    # simulation window (ps)
Points  = 2**13

fig, axs = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)

parameters = dict(center_wavelength_nm=pulseWL, time_window_ps=Window, 
                  fwhm_ps=FWHM, GDD=GDD, TOD=TOD, npts=Points,
                  frep_MHz=100, power_is_avg=False, epp=EPP)

# create the pulses:
for pulse_type in ('sech', 'gaussian', 'sinc'):
    print (pulse_type)
    pulse = nlse.pulse.Pulse(pulse_type=pulse_type, **parameters)

    def dB(x):
        return 10 * np.log10(np.abs(x)**2)
        
    axs[0,0].plot(pulse.f_THz, np.abs(pulse.aw)**2,  label=pulse_type)
    axs[0,1].plot(pulse.t_ps,  np.abs(pulse.at)**2,  label=pulse_type)
    axs[1,0].plot(pulse.f_THz, dB(pulse.aw), label=pulse_type)
    axs[1,1].plot(pulse.t_ps,  dB(pulse.at), label=pulse_type)

f0 = pulse.centerfrequency_THz

# upper left
axs[0,0].set_xlabel('Frequency (THz)')
axs[0,0].set_ylabel('Amplitude, linear scale (J*Hz)')
axs[0,0].set_xlim(f0-2/FWHM, f0+2/FWHM)

# lower left
axs[1,0].set_xlabel('Frequency (THz)')
axs[1,0].set_ylabel('Amplitude, log scale (dB(J*Hz))')
axs[1,0].set_xlim(f0-20/FWHM, f0+20/FWHM)
axs[1,0].set_ylim(-400, 20)

# upper right
axs[0,1].set_ylabel('Amplitude, linear scale (J/sec)')
axs[0,1].set_xlabel('Frequency (THz)')
axs[0,1].set_xlim(-FWHM*5, FWHM*5)

# lower right
axs[1,1].set_ylabel('Amplitude, log scale (dB(J/sec))')
axs[1,1].set_xlabel('Time (ps)')
axs[1,1].set_xlim(-FWHM*20, FWHM*20)
axs[1,1].set_ylim(-350, 50)

for ax in axs.ravel():
    ax.grid(color='k', alpha=0.1)
    ax.legend()
    
plt.show()






