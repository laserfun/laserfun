"""
This script propagates a pulse through fiber with the D and D_slope parameters
entered. It determines where the shortest pulse is and compares the time-domain
profile of this pulse to the transform limit.
"""

import laserfun as lf
import matplotlib.pyplot as plt
import numpy as np


# Pulse parameters:
pulseWL = 1550  # pulse central wavelength (nm)
FWHM = 0.8  # pulse duration (ps)
EPP = 300e-12  # Energy per pulse (J)

# # Thorlabs PMHN3N (Normal dispersion)
# disp = -2.4    # fiber dispersion in ps/(nm km)
# slope = 0.025  # dispersion slope in ps/(nm^2 km)
# length = 5     # length of first fiber in meters
# alpha = 0      # loss (dB/cm)
# gamma = 10.8   # nonlinearity (1/(W km))

# Thorlabs PMHN5
disp = 5  # fiber dispersion in ps/(nm km)
slope = 0.025  # dispersion slope in ps/(nm^2 km)
length = 2.02  # length of second fiber in meters
alpha = 0  # loss (dB/cm)
gamma = 10.8  # nonlinearity (1/(W km))

# # Thorlabs PMHN1
# disp = 1       # fiber dispersion in ps/(nm km)
# slope = 0.025  # dispersion slope in ps/(nm^2 km)
# length = 1.8   # length of second fiber in meters
# alpha = 0      # loss (dB/cm)
# gamma = 10.8   # nonlinearity (1/(W km))

# # PM1550
# disp = 18     # fiber dispersion in ps/(nm km)
# slope = 0.025  # dispersion slope in ps/(nm^2 km)
# length = 1.8   # length of second fiber in meters
# alpha = 0      # loss (dB/cm)
# gamma = 0.78   # nonlinearity (1/(W km))

# initial pulse dispersion:
GDD = 0.0  # Group delay dispersion (ps^2)
TOD = 0.0  # Third order dispersion (ps^3)

# simulation parameters
Window = 20.0  # simulation window (ps)
Steps = 100  # simulation steps
Points = 2**12  # simulation points
rtol = 1e-4  # relative error for NLSE integrator
atol = 1e-4  # absolute error

Raman = True  # Enable Raman effect?
Steep = True  # Enable self steepening?

level = 0.4  # level relative to maximum to evaluate pulse duration

# ----- END OF PARAMETERS -----

ps_nm_km = -(pulseWL**2) / (2 * np.pi * 2.9979246e5)  # conversion for D2
ps2_nm2 = pulseWL**4 / (4 * np.pi**2 * 2.9979246e5**2)  # conversion for D3
ps2_nm = pulseWL**3 / (2 * np.pi**2 * 2.9979246e5**2)

# fiber1
beta2 = disp * ps_nm_km  # (ps^2/km)
beta3 = slope * ps2_nm2 + disp * ps2_nm  # (ps^3/km)


# create the pulse
pulse_in = lf.Pulse(
    pulse_type="sech",
    fwhm_ps=FWHM,
    center_wavelength_nm=pulseWL,
    time_window_ps=Window,
    GDD=GDD,
    TOD=TOD,
    npts=Points,
    power_is_avg=False,
    epp=EPP,
)

# create the fiber
fiber = lf.Fiber(
    length,
    center_wl_nm=pulseWL,
    dispersion_format="GVD",
    dispersion=(beta2 * 1e-3, beta3 * 1e-3),
    gamma_W_m=gamma * 1e-3,
    loss_dB_per_m=alpha * 100,
)

print("Propagation in fiber...")
results = lf.NLSE(
    pulse_in,
    fiber,
    raman=Raman,
    shock=Steep,
    nsaves=Steps,
    rtol=rtol,
    atol=atol,
    print_status=False,
    custom_raman="dudley",
)

pulse1 = results.pulse_out


print("Plotting results...")
# plot the results
fig1, axs1 = results.plot(tlim=(-2, 2), show=False, wavelength=True, units="dBm/nm")

# calculate pulse durations
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)


def calc_width(t_ps, at, level=level):
    def find_roots(x, y):
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)

    it = np.abs(at) ** 2
    it = it / np.max(it)
    roots = find_roots(t_ps, it - level)
    width = np.max(roots) - np.min(roots)
    return width


widths = np.zeros_like(results.z)

for n, at in enumerate(results.AT):
    widths[n] = calc_width(pulse_in.t_ps, at)

ind = np.argmin(widths)
ax3.plot(results.z, widths, label="Pulse duration")
ax3.plot(
    results.z[ind],
    widths[ind],
    "o",
    color="C1",
    label="Shortest pulse: %.3f ps\n %.3f meters" % (widths[ind], results.z[ind]),
)

shortest = pulse1.create_cloned_pulse()
shortest.at = results.AT[ind]

ax3.legend(labelcolor="linecolor")
ax3.grid(alpha=0.2, color="k")
ax3.set_xlabel("Propagation length (meters)")
ax3.set_ylabel("Pulse duration (ps)")

# plots the linear-scale time domain:
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)

pulse1 = results.pulse_out
tl = shortest.transform_limit()

for pulse, label in zip(
    (pulse_in, shortest, tl), ("Input pulse", "Shortest pulse", "Transform limit")
):

    width = pulse.calc_width(level=level) * 1e3
    (l,) = ax4.plot(
        pulse.t_ps, np.abs(pulse.at) ** 2, label=label + ": %.1f fs" % (width)
    )

ax4.set_xlim(-2, 2)
ax4.legend(labelcolor="linecolor")
ax4.grid(alpha=0.2, color="k")
ax4.set_xlabel("Time (ps)")
ax4.set_ylabel("Intensity")

print("Done.")
fig4.savefig("HNLF propagation.png", dpi=250)

plt.show()
