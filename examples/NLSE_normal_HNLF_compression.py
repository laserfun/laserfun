"""
This script propagates a pulse through fiber with the D and D_slope parameters
entered. It then propagates the pulse through a second length of fiber and
determines where the shortest pulse is and compares the time-domain profile
of this pulse to the transform limit. The script was created to propagate a
pulse through normal dispersion highly nonlinear fiber (HNLF) to generate
additional bandwidth and then use anomalous dispersion fiber (PM1550) to
provide dispersion to compress the pulse.
"""

import laserfun as lf
import matplotlib.pyplot as plt
import numpy as np


# Pulse parameters:
pulseWL = 1550  # pulse central wavelength (nm)
FWHM = 0.8  # pulse duration (ps)
EPP = 300e-12  # Energy per pulse (J)

# Fiber 1:
disp1 = -2.4  # fiber dispersion in ps/(nm km)
slope1 = 0.025  # dispersion slope in ps/(nm^2 km)
length1 = 5  # length of first fiber in meters
alpha1 = 0  # loss (dB/cm)
gamma1 = 10.8  # nonlinearity (1/(W km))

# - second fiber:
disp2 = 18  # fiber dispersion in ps/(nm km)
slope2 = 0.025  # dispersion slope in ps/(nm^2 km)
length2 = 1.8  # length of second fiber in meters
alpha2 = 0  # loss (dB/cm)
gamma2 = 0.78  # nonlinearity (1/(W km))

loss_between_sections_dB = 1

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
custom_raman = "dudley"

level = 0.4  # level relative to maximum to evaluate pulse duration

# ----- END OF PARAMETERS -----

ps_nm_km = -(pulseWL**2) / (2 * np.pi * 2.9979246e5)  # conversion for D2
ps2_nm2 = pulseWL**4 / (4 * np.pi**2 * 2.9979246e5**2)  # conversion for D3
ps2_nm = pulseWL**3 / (2 * np.pi**2 * 2.9979246e5**2)

# fiber1
beta2 = disp1 * ps_nm_km  # (ps^2/km)
beta3 = slope1 * ps2_nm2 + disp1 * ps2_nm  # (ps^3/km)

beta3b = (pulseWL**4/(4*np.pi**2*2.9979246e5**2))*slope1 \
        - (pulseWL/(np.pi*2.9979246e5))*beta2
        
print(beta3, beta3b)

print(beta2, -(pulseWL**2) / (2 * np.pi * 2.9979246e5) * disp1)
        

# fiber 2
beta22 = disp2 * ps_nm_km  # (ps^2/km)
beta32 = slope2 * ps2_nm2 + disp2 * ps2_nm  # (ps^3/km)

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
fiber1 = lf.Fiber(
    length1,
    center_wl_nm=pulseWL,
    dispersion_format="GVD",
    dispersion=(beta2 * 1e-3, beta3 * 1e-3),
    gamma_W_m=gamma1 * 1e-3,
    loss_dB_per_m=alpha1 * 100,
)

print("Propagation in fiber1...")
results1 = lf.NLSE(
    pulse_in,
    fiber1,
    raman=Raman,
    shock=Steep,
    nsaves=Steps,
    rtol=rtol,
    atol=atol,
    print_status=False,
    custom_raman=custom_raman,
)

# second fiber
fraction = 10 ** (-loss_between_sections_dB * 0.1)
pulse1 = results1.pulse_out
pulse1.aw = pulse1.aw * np.sqrt(fraction)

# create the fiber!
fiber2 = lf.Fiber(
    length2,
    center_wl_nm=pulseWL,
    dispersion_format="GVD",
    dispersion=(beta22 * 1e-3, beta32 * 1e-3),
    gamma_W_m=gamma2 * 1e-3,
    loss_dB_per_m=alpha2 * 100,
)

print("Propagation in fiber2...")
results2 = lf.NLSE(
    pulse1,
    fiber2,
    raman=Raman,
    shock=Steep,
    nsaves=Steps,
    rtol=rtol,
    atol=atol,
    print_status=False,
    custom_raman=custom_raman,
)
pulse2 = results2.pulse_out

print("Plotting results...")
# plot the results
fig1, axs1 = results1.plot(flim=(188, 200), tlim=(-2, 2), show=False)
fig2, axs2 = results2.plot(flim=(188, 200), tlim=(-2, 2), show=False)


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


widths1 = np.zeros_like(results1.z)
widths2 = np.zeros_like(results2.z)
widths2 = np.zeros_like(results2.z)

for n, at in enumerate(results1.AT):
    widths1[n] = calc_width(pulse1.t_ps, at)

for n, at in enumerate(results2.AT):
    widths2[n] = calc_width(pulse2.t_ps, at)

ind = np.argmin(widths2)
ax3.plot(results1.z, widths1, label="fiber1")
ax3.plot(results2.z + np.max(results1.z), widths2, label="fiber2")
ax3.plot(
    results2.z[ind] + np.max(results1.z),
    widths2[ind],
    "o",
    color="C1",
    label="Shortest pulse: %.3f ps\n %.3f meters into fiber2"
    % (widths2[ind], results2.z[ind]),
)

shortest = pulse2.create_cloned_pulse()
shortest.at = results2.AT[ind]

ax3.legend(labelcolor="linecolor")
ax3.grid(alpha=0.2, color="k")
ax3.set_xlabel("Propagation length (meters)")
ax3.set_ylabel("Pulse duration (ps)")

axs2[1, 0].axhline(results2.z[ind] * 1e3, color="r", ls="dashed")
axs2[1, 1].axhline(results2.z[ind] * 1e3, color="r", ls="dashed")

# plots the linear-scale time domain:
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)

pulse2 = results2.pulse_out
tl = shortest.transform_limit()

# for pulse, label in zip((pulse_in, pulse1, pulse2, shortest, tl), ('Input pulse', 'After fiber1', 'After fiber2', 'Shortest pulse in fiber2', 'Transform limit')):
for pulse, label in zip(
    (pulse_in, shortest, tl),
    ("Input pulse", "Shortest pulse in fiber2", "Transform limit"),
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

if __name__ == "__main__":
    fig4.savefig("Compressing in HNLF.png", dpi=250)
    plt.show()
