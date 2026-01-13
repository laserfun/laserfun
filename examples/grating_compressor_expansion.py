
import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf
from laserfun import tools
from laserfun.pulse import c_nmps

# This example shows the effect of different dispersions on a pulse
# compressed by a grating compressor. It also compares to the full
# theory of the grating compressor.

# Parameters
wl = 1545.0      # nm (Center wavelength)
lines = 1000     # l/mm (Grating constant)
separation = 4e-3 # 4 mm perpendicular separation
fwhm_ps = 0.1    # 100 fs pulse width

# Initialize Compressor
comp = tools.TreacyCompressor(lines_per_mm=lines, littrow_wavelength_nm=wl)

# Calculate Dispersion Coefficients (in ps^2, ps^3, ps^4)
disp = comp.calc_dispersion(wl, separation, order=[2, 3, 4])
gdd_ps2, tod_ps3, fod_ps4 = disp

# Setup Pulses
npts = 2**15
time_window = 80.0 # ps (Increased for finer spectral resolution)
config = {
    'pulse_type': 'gaussian', 
    'fwhm_ps': fwhm_ps, 
    'center_wavelength_nm': wl,
    'time_window_ps': time_window, 
    'npts': npts
}

pulses = {
    'Original':     lf.Pulse(**config),
    'GDD only':     lf.Pulse(**config),
    'GDD + TOD':    lf.Pulse(**config),
    'GDD+TOD+FOD':  lf.Pulse(**config),
    'Complete':     lf.Pulse(**config)
}

# Apply approximations
pulses['GDD only'].chirp_pulse_W(GDD=gdd_ps2)
pulses['GDD + TOD'].chirp_pulse_W(GDD=gdd_ps2, TOD=tod_ps3)
pulses['GDD+TOD+FOD'].chirp_pulse_W(GDD=gdd_ps2, TOD=tod_ps3, FOD=fod_ps4)

# Apply full theory
comp.apply_phase_to_pulse(separation, pulses['Complete'])

# Plotting
plt.figure(figsize=(12, 5))

# Left Panel: Spectral Phase
plt.subplot(1, 2, 1)
freq = pulses['Original'].f_THz
pulse_center_f = c_nmps / wl
mask = np.abs(freq - pulse_center_f) < 15 # Zoom in on bandwidth

for label, p in pulses.items():
    if label == 'Original': continue
    
    # Calculate phase relative to the original pulse to remove
    # any intrinsic grid-phase or time-offsets.
    # Use a small epsilon to avoid divide-by-zero warnings in low-intensity regions.
    denom = pulses['Original'].aw
    mask_nonzero = np.abs(denom) > 1e-15
    rel_aw = np.ones_like(p.aw, dtype=complex)
    rel_aw[mask_nonzero] = p.aw[mask_nonzero] / denom[mask_nonzero]
    
    phase = np.unwrap(np.angle(rel_aw))
    
    # Anchor at center frequency
    w0_idx = np.argmin(np.abs(freq - pulse_center_f))
    phase -= phase[w0_idx]
    
    alpha = 1.0 if label == 'Complete' else 0.7
    ls = '-' if label == 'Complete' else '--'
    plt.plot(freq[mask], phase[mask], label=label, alpha=alpha, linestyle=ls)

plt.xlabel('Frequency (THz)')
plt.ylabel('Spectral Phase (rad)')
plt.title('Spectral Phase Comparison')
plt.legend()
plt.grid(alpha=0.3)

# --- Right Panel: Temporal Pulse ---
plt.subplot(1, 2, 2)
t = pulses['Original'].t_ps

for label, p in pulses.items():
    if label == 'Original': continue # Skip unchirped pulse for better scale
    alpha = 1.0 if label == 'Complete' else 0.6
    plt.plot(t, p.it, label=label, alpha=alpha)
    
# Calculate intelligent x-limits based on expected stretching
# Stretched width roughly ~ 4 * |GDD| / T0
t0_ps = fwhm_ps / 1.665 # Gaussian factor for field 1/e
stretched_fwhm = fwhm_ps * np.sqrt(1 + (gdd_ps2 / t0_ps**2)**2)
x_limit = 3.0 * stretched_fwhm

plt.xlim(-x_limit, x_limit)
plt.xlabel('Time (ps)')
plt.ylabel('Intensity (W)')
plt.title('Temporal Pulse Shape')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

if __name__ == "__main__":
    plt.show()
