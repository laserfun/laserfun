
import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf
from laserfun import tools

# Parameters
l_mm = 1000
centerWL = 1545.0
separation = np.linspace(0.0, .30, 4)

# compare to pm1550
ref_dispersion = 18.0 #ps/nm/km
ref_slope = 0.06 #ps/nm^2/km
l1 = 50  # meters
l2 = 500 # meters

# Initialize Compressor
comp = tools.TreacyCompressor(lines_per_mm=l_mm, littrow_wavelength_nm=centerWL)

# Calculate dispersions in engineering units (ps/nm, ps/nm^2)
# and also in beta-units (ps^n) for the plots
D_vals, S_vals = comp.calc_dispersion_D(centerWL, separation)
beta2_ps2, beta3_ps3 = comp.calc_dispersion(centerWL, separation, order=[2, 3])
print('Angle (deg): ', comp.g*180/np.pi)
print("Separation (m):", separation)
print("GDD (ps^2):", beta2_ps2)
print("TOD (ps^3):", beta3_ps3)
print("D (ps/nm):", D_vals)
print("S (ps/nm^2):", S_vals)

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# Left column: Beta coefficients (ps^n)
# Plot GDD 
axs[0,0].plot(1e3*separation, beta2_ps2, label='Grating dispersion')
axs[0,0].set_ylabel('GDD ($ps^2$)')
axs[0,0].set_title('Group Velocity Dispersion')

# Plot TOD
axs[1,0].plot(1e3*separation, beta3_ps3, label='Grating dispersion')
axs[1,0].set_ylabel('TOD ($ps^3$)')
axs[1,0].set_title('Third Order Dispersion')

# Right column: Engineering units (ps/nm, ps/nm^2)
# Plot D
axs[0,1].plot(1e3*separation, D_vals, label='Grating dispersion')
axs[0,1].set_ylabel('Dispersion D (ps/nm)')
axs[0,1].set_title('Dispersion Parameter')

# Plot S
axs[1,1].plot(1e3*separation, S_vals, label='Grating dispersion')
axs[1,1].set_ylabel('Dispersion Slope S ($ps/nm^2$)')
axs[1,1].set_title('Dispersion Slope')

# Comparisons on GDD plot
for i, l in enumerate((l1, l2)):
    color = 'C3' if i == 0 else 'C4'  # Different colors for 20m vs 200m
    beta2 = lf.tools.D2_to_beta2(centerWL, ref_dispersion)*l*1e-3
    beta3 = lf.tools.D3_to_beta3(centerWL, D2=ref_dispersion, D3=ref_slope)*l*1e-3
    axs[0,1].axhline(ref_dispersion*l*1e-3, alpha=0.5, color=color, label='%.0f m of %.2f ps/nm/km fiber = %.2f ps/nm' % (l, ref_dispersion, ref_dispersion*l*1e-3))
    axs[1,1].axhline(ref_slope*l*1e-3,      alpha=0.5, color=color, label='%.0f m of %.3f ps/nm/km fiber = %.4f ps/nm' % (l, ref_slope, ref_slope*l*1e-3))
    axs[0,0].axhline(beta2,                 alpha=0.5, color=color, label='%.0f m of %.2f ps/nm^2/km fiber = %.2f ps/nm^2' % (l, ref_dispersion, beta2))
    axs[1,0].axhline(beta3,                 alpha=0.5, color=color, label='%.0f m of %.3f ps/nm^2/km fiber = %.4f ps^3' % (l, ref_slope, beta3))

for ax in axs.flatten():
    ax.legend(fontsize=9)
    ax.set_xlabel('Grating separation (mm)')
    ax.grid(alpha=0.2)
    
plt.suptitle('Compressor dispersion for %.0f lines/mm, center wavelength = %.1f nm, Littrow angle = %.1f deg' % (l_mm, centerWL, comp.g*180/np.pi))
plt.tight_layout()
plt.savefig('Compressor GDD and TOD.png', dpi=200)

if __name__ == "__main__":
    plt.show()