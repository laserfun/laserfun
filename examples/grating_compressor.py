import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf
from laserfun import tools

# Parameters
l_mm = 1000
angle = 50.5789
centerWL = 1545

# Initialize Compressor
comp = tools.TreacyCompressor(l_mm, angle)

# Calculate diffraction angle
theta = comp.calc_theta(centerWL)
print(f"Diffraction angle (theta): {theta*180/np.pi:.2f} deg")

separation = np.linspace(0.0, .30, 4)

# Calculate dispersions
# Note: calc_compressor_gdd and HOD return values in SI units (s^2, s^3)
# We need to scale them for plotting.

gdd = comp.calc_compressor_gdd(centerWL, separation)
tod = comp.calc_compressor_HOD(centerWL, separation, 3)

print("Separation (m):", separation)
print("GDD (s^2):", gdd)
print("TOD (s^3):", tod)

# Convert to D and S
# tools functions expect beta in ps^2/km and ps^3/km
# gdd is in s^2, tod is in s^3.
# 1 s^2 = 1e24 ps^2. 
# But the functions expect per km. Since these are TOTAL values for the compressor (not per length),
# we can treat them as if the length was 1 km to get the total D and S "values" (units ps/nm).
# Effectively: D(total) = beta2(total) * conversion_factor
# The conversion functions are linear in beta, so this works.

beta2_ps2 = gdd * 1e24
beta3_ps3 = tod * 1e36

# Calculate D (ps/nm) and S (ps/nm^2)
D_vals = lf.tools.beta2_to_D2(centerWL, beta2_ps2)
S_vals = lf.tools.beta3_to_D3(centerWL, beta3_ps3, beta2=beta2_ps2)

print("D (ps/nm):", D_vals)
print("S (ps/nm^2):", S_vals)

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# Left column: Beta coefficients (ps^n)
# Plot GDD 
axs[0,0].plot(1e3*separation, beta2_ps2)
axs[0,0].set_ylabel('GDD ($ps^2$)')
axs[0,0].set_title('Group Velocity Dispersion')

# Plot TOD
axs[1,0].plot(1e3*separation, beta3_ps3)
axs[1,0].set_ylabel('TOD ($ps^3$)')
axs[1,0].set_title('Third Order Dispersion')

# Right column: Engineering units (ps/nm, ps/nm^2)
# Plot D
axs[0,1].plot(1e3*separation, D_vals)
axs[0,1].set_ylabel('Dispersion D (ps/nm)')
axs[0,1].set_title('Dispersion Parameter')

# Plot S
axs[1,1].plot(1e3*separation, S_vals)
axs[1,1].set_ylabel('Dispersion Slope S ($ps/nm^2$)')
axs[1,1].set_title('Dispersion Slope')

# Comparisons on GDD plot
axs[0,0].axhline(-0.051, alpha=0.3, color='r', label='20 m of -2 ps/nm/km fiber')
axs[0,0].axhline(-0.51,  alpha=0.3, color='b', label='200 m of -2 ps/nm/km fiber')
axs[0,0].legend(fontsize=9)


for ax in axs.flatten():
    ax.set_xlabel('Grating separation (mm)')
    ax.grid(alpha=0.2)
    
plt.suptitle('Compressor dispersion for %.0f lines/mm and angle = %.1f degrees' % (l_mm, angle))
plt.tight_layout()
plt.savefig('Compressor GDD and TOD.png', dpi=200)
plt.show()