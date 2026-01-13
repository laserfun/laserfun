"""
This example demonstrates the effect of dispersion parameter D and 
dispersion slope S on pulses of different durations.
"""

import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf

# Pulse parameters
center_wavelength_nm = 1550
time_window_ps = 10.0
npts = 2**13

# Pulse durations to test (in femtoseconds)
pulse_durations_fs = [50, 100, 200]

# D values to test (in ps/nm)
D_values = [0.005, 0.01, 0.02]

# S values to test (in ps/nm²)
S_values = [0.005, 0.01, 0.02]

# Create figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Color cycle for different dispersion values
colors = ['C0', 'C1', 'C2']

# Loop through pulse durations (rows)
for row, fwhm_fs in enumerate(pulse_durations_fs):
    fwhm_ps = fwhm_fs * 1e-3
    
    # Left column: D effects
    ax_D = axs[row, 0]
    
    # Plot original pulse
    p_orig = lf.Pulse(
        pulse_type='gaussian',
        fwhm_ps=fwhm_ps,
        center_wavelength_nm=center_wavelength_nm,
        time_window_ps=time_window_ps,
        npts=npts
    )
    ax_D.plot(p_orig.t_ps, p_orig.it, color='k', ls='dashed', linewidth=1.5, label='Original', alpha=0.7)
    
    # Plot pulses with different D values
    for i, D in enumerate(D_values):
        # Convert D to GDD (beta2)
        gdd = lf.tools.D2_to_beta2(center_wavelength_nm, D)
        
        p = lf.Pulse(
            pulse_type='gaussian',
            fwhm_ps=fwhm_ps,
            center_wavelength_nm=center_wavelength_nm,
            time_window_ps=time_window_ps,
            npts=npts,
            GDD=gdd
        )
        ax_D.plot(p.t_ps, p.it, color=colors[i], label=f'D = {D} ps/nm', alpha=0.8)
    
    ax_D.set_ylabel('Intensity (W)')
    ax_D.set_title(f'{fwhm_fs} fs pulse - Dispersion D effect')
    ax_D.legend(fontsize=8)
    ax_D.grid(alpha=0.3)
    
    # Set x-limits based on pulse duration
    x_limit = max(2.0, fwhm_ps * 20)
    ax_D.set_xlim(-x_limit, x_limit)
    
    # Right column: S effects
    ax_S = axs[row, 1]
    
    # Plot original pulse
    ax_S.plot(p_orig.t_ps, p_orig.it, 'k--', linewidth=2, label='Original', alpha=0.7)
    
    # Plot pulses with different S values
    for i, S in enumerate(S_values):
        # Convert S to TOD (beta3)
        # Note: S depends on both D and beta3, so we need to provide D=0 for pure TOD effect
        tod = lf.tools.D3_to_beta3(center_wavelength_nm, D2=0, D3=S)
        
        p = lf.Pulse(
            pulse_type='gaussian',
            fwhm_ps=fwhm_ps,
            center_wavelength_nm=center_wavelength_nm,
            time_window_ps=time_window_ps,
            npts=npts,
            TOD=tod
        )
        ax_S.plot(p.t_ps, p.it, color=colors[i], label=f'S = {S} ps/nm²', alpha=0.8)
    
    ax_S.set_ylabel('Intensity (W)')
    ax_S.set_title(f'{fwhm_fs} fs pulse - Dispersion Slope S effect')
    ax_S.legend(fontsize=8)
    ax_S.grid(alpha=0.3)
    ax_S.set_xlim(-x_limit, x_limit)

# Set x-labels only on bottom row
for ax in axs[2, :]:
    ax.set_xlabel('Time (ps)')

plt.suptitle('Effect of Dispersion D and Slope S on Pulses of Different Durations', fontsize=14)
plt.tight_layout()

if __name__ == "__main__":
    plt.show()
