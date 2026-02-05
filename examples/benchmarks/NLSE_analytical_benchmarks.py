"""
NLSE Analytical Benchmarks

This example validates the laserfun NLSE solver against known analytical 
solutions for nonlinear pulse propagation:

1. Fundamental soliton (N=1) - maintains shape during propagation
2. Higher-order soliton (N=2) - periodic breathing with period z₀ = (π/2)·L_D
3. Pure dispersion - Gaussian broadening τ(z)/τ₀ = √(1 + (z/L_D)²)
4. Pure SPM - temporal shape unchanged, spectrum broadens

References:
- Agrawal, "Nonlinear Fiber Optics" (5th ed.), Chapter 5
- Dudley & Taylor, "Supercontinuum Generation in Optical Fibers" (2010)
"""

import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf

# Physical constants
c_mks = 299792458.0  # m/s


def calc_soliton_parameters(pulse, fiber):
    """Calculate soliton number N, dispersion length L_D, and soliton period z₀.
    
    For sech pulses: T₀ = FWHM / 1.763
    L_D = T₀² / |β₂|
    z₀ = (π/2) × L_D  (soliton period)
    N = √(γ P₀ T₀² / |β₂|)
    """
    # For sech pulse, T₀ = FWHM / 1.763
    T0_ps = pulse.calc_width(level=0.5) / 1.763
    T0_s = T0_ps * 1e-12
    
    # Get β₂ from fiber (first element of betas is β₂ in ps²/m)
    beta2_ps2_m = fiber.betas[0]
    beta2_s2_m = beta2_ps2_m * 1e-24
    
    # Peak power
    P0 = np.max(pulse.it)
    
    # Dispersion length
    L_D = T0_s**2 / np.abs(beta2_s2_m)
    
    # Soliton period
    z0 = (np.pi / 2) * L_D
    
    # Soliton number
    gamma = fiber.gamma
    N = np.sqrt(gamma * P0 * T0_s**2 / np.abs(beta2_s2_m))
    
    return N, L_D, z0, T0_ps


def create_soliton_pulse(N, T0_ps, wavelength_nm, beta2_ps2_m, gamma_W_m):
    """Create a sech pulse with specified soliton number N.
    
    Given N, T₀, β₂, and γ, we solve for the required peak power:
    P₀ = N² |β₂| / (γ T₀²)
    """
    T0_s = T0_ps * 1e-12
    beta2_s2_m = beta2_ps2_m * 1e-24
    
    # Required peak power for soliton order N
    P0 = N**2 * np.abs(beta2_s2_m) / (gamma_W_m * T0_s**2)
    
    # FWHM from T₀ (for sech: FWHM = 1.763 × T₀)
    fwhm_ps = T0_ps * 1.763
    
    # Create pulse - need to set energy to achieve correct peak power
    # For sech: E = 2 * P0 * T0 (exact integral of sech²)
    epp = 2 * P0 * T0_s
    
    pulse = lf.Pulse(
        pulse_type='sech',
        fwhm_ps=fwhm_ps,
        center_wavelength_nm=wavelength_nm,
        time_window_ps=max(50 * fwhm_ps, 10),
        npts=2**13,
        epp=epp,
    )
    
    return pulse


def benchmark_fundamental_soliton():
    """Benchmark 1: N=1 soliton should maintain its shape."""
    print("\n" + "="*60)
    print("Benchmark 1: Fundamental Soliton (N=1) Stability")
    print("="*60)
    
    # Parameters
    wavelength_nm = 1550
    T0_ps = 0.1  # 100 fs T₀
    beta2_ps2_m = -0.020  # anomalous dispersion (ps²/m)
    gamma_W_m = 1.0  # 1/(W·m)
    
    # Create N=1 soliton
    pulse = create_soliton_pulse(N=1, T0_ps=T0_ps, wavelength_nm=wavelength_nm,
                                  beta2_ps2_m=beta2_ps2_m, gamma_W_m=gamma_W_m)
    
    fiber = lf.Fiber(
        length=0.05,  # 5 cm (several L_D)
        center_wl_nm=wavelength_nm,
        dispersion=[beta2_ps2_m],
        gamma_W_m=gamma_W_m,
        loss_dB_per_m=0,
    )
    
    # Calculate parameters
    N, L_D, z0, _ = calc_soliton_parameters(pulse, fiber)
    print(f"Soliton number N = {N:.3f}")
    print(f"Dispersion length L_D = {L_D*1e3:.2f} mm")
    print(f"Soliton period z₀ = {z0*1e3:.2f} mm")
    print(f"Propagation = {fiber.length/L_D:.1f} × L_D")
    
    # Propagate
    results = lf.NLSE(pulse, fiber, nsaves=50, print_status=False,
                      raman=False, shock=False, atol=1e-8, rtol=1e-8)
    
    # Analyze: peak power should stay constant
    z, f, t, AW, AT = results.get_results(data_type='intensity')
    peak_powers = np.max(AT, axis=1)
    initial_peak = peak_powers[0]
    
    peak_variation = (np.max(peak_powers) - np.min(peak_powers)) / initial_peak
    
    print(f"\nResults:")
    print(f"Peak power variation: {peak_variation*100:.2f}%")
    
    success = peak_variation < 0.02  # < 2% variation
    print(f"✓ PASSED" if success else "✗ FAILED")
    
    return results, success, peak_variation


def benchmark_higher_order_soliton():
    """Benchmark 2: N=2 soliton should return to initial shape at z = z₀."""
    print("\n" + "="*60)
    print("Benchmark 2: Higher-Order Soliton (N=2) Period")
    print("="*60)
    
    # Parameters
    wavelength_nm = 1550
    T0_ps = 0.1
    beta2_ps2_m = -0.020
    gamma_W_m = 1.0
    
    # Create N=2 soliton
    pulse = create_soliton_pulse(N=2, T0_ps=T0_ps, wavelength_nm=wavelength_nm,
                                  beta2_ps2_m=beta2_ps2_m, gamma_W_m=gamma_W_m)
    
    # Calculate soliton period
    T0_s = T0_ps * 1e-12
    beta2_s2_m = beta2_ps2_m * 1e-24
    L_D = T0_s**2 / np.abs(beta2_s2_m)
    z0 = (np.pi / 2) * L_D
    
    # Propagate for exactly one soliton period
    fiber = lf.Fiber(
        length=z0,
        center_wl_nm=wavelength_nm,
        dispersion=[beta2_ps2_m],
        gamma_W_m=gamma_W_m,
        loss_dB_per_m=0,
    )
    
    N_calc, _, _, _ = calc_soliton_parameters(pulse, fiber)
    print(f"Soliton number N = {N_calc:.3f}")
    print(f"Dispersion length L_D = {L_D*1e3:.2f} mm")
    print(f"Soliton period z₀ = {z0*1e3:.2f} mm")
    
    results = lf.NLSE(pulse, fiber, nsaves=100, print_status=False,
                      raman=False, shock=False, atol=1e-8, rtol=1e-8)
    
    # Compare input and output intensity profiles
    z, f, t, AW, AT = results.get_results(data_type='intensity')
    
    initial_profile = AT[0] / np.max(AT[0])
    final_profile = AT[-1] / np.max(AT[-1])
    
    # Correlation coefficient
    correlation = np.corrcoef(initial_profile, final_profile)[0, 1]
    
    # Peak power ratio
    peak_ratio = np.max(AT[-1]) / np.max(AT[0])
    
    print(f"\nResults:")
    print(f"Correlation (input vs output): {correlation:.4f}")
    print(f"Peak power ratio: {peak_ratio:.3f}")
    
    success = correlation > 0.98 and 0.9 < peak_ratio < 1.1
    print(f"✓ PASSED" if success else "✗ FAILED")
    
    return results, success, correlation, peak_ratio


def benchmark_linear_dispersion():
    """Benchmark 3: Pure dispersion (γ=0) should follow analytical broadening."""
    print("\n" + "="*60)
    print("Benchmark 3: Pure Dispersion (Linear) Broadening")
    print("="*60)
    
    # Parameters
    wavelength_nm = 1550
    fwhm_ps = 0.1  # 100 fs FWHM
    beta2_ps2_m = -0.020
    
    # For Gaussian: T₀ = FWHM / (2√(ln2)) ≈ FWHM / 1.665
    T0_ps = fwhm_ps / (2 * np.sqrt(np.log(2)))
    T0_s = T0_ps * 1e-12
    beta2_s2_m = beta2_ps2_m * 1e-24
    L_D = T0_s**2 / np.abs(beta2_s2_m)
    
    print(f"Gaussian T₀ = {T0_ps*1e3:.2f} fs")
    print(f"Dispersion length L_D = {L_D*1e3:.2f} mm")
    
    # Propagate for 5 L_D
    fiber_length = 5 * L_D
    
    pulse = lf.Pulse(
        pulse_type='gaussian',
        fwhm_ps=fwhm_ps,
        center_wavelength_nm=wavelength_nm,
        time_window_ps=20,
        npts=2**13,
        epp=1e-12,  # Low energy - negligible nonlinearity
    )
    
    fiber = lf.Fiber(
        length=fiber_length,
        center_wl_nm=wavelength_nm,
        dispersion=[beta2_ps2_m],
        gamma_W_m=0,  # No nonlinearity!
        loss_dB_per_m=0,
    )
    
    results = lf.NLSE(pulse, fiber, nsaves=51, print_status=False,
                      raman=False, shock=False, atol=1e-8, rtol=1e-8)
    
    # Measure width at each z-position
    z, f, t, AW, AT = results.get_results(data_type='intensity')
    
    measured_widths = []
    
    # Use second moment (RMS width) - mathematically robust
    # For a Gaussian: FWHM = 2*sqrt(2*ln(2)) * sigma_rms ≈ 2.355 * sigma_rms
    for i in range(len(z)):
        profile = AT[i]
        # Calculate RMS width: σ² = ∫t²|A|²dt / ∫|A|²dt
        total_power = np.sum(profile)
        t_mean = np.sum(t * profile) / total_power
        sigma_rms = np.sqrt(np.sum((t - t_mean)**2 * profile) / total_power)
        # Convert RMS width to FWHM for Gaussian
        width = sigma_rms * 2 * np.sqrt(2 * np.log(2))
        measured_widths.append(width)
    
    measured_widths = np.array(measured_widths)
    initial_width = measured_widths[0]
    
    # Analytical prediction: τ(z)/τ₀ = √(1 + (z/L_D)²)
    z_over_LD = z / L_D
    analytical_ratio = np.sqrt(1 + z_over_LD**2)
    analytical_widths = initial_width * analytical_ratio
    
    # Calculate error
    relative_error = np.abs(measured_widths - analytical_widths) / analytical_widths
    max_error = np.max(relative_error)
    mean_error = np.mean(relative_error)
    
    print(f"\nResults:")
    print(f"Initial width: {initial_width*1e3:.2f} fs")
    print(f"Final width (measured): {measured_widths[-1]*1e3:.2f} fs")
    print(f"Final width (analytical): {analytical_widths[-1]*1e3:.2f} fs")
    print(f"Mean relative error: {mean_error*100:.2f}%")
    print(f"Max relative error: {max_error*100:.2f}%")
    
    success = max_error < 0.05  # < 5% error
    print(f"✓ PASSED" if success else "✗ FAILED")
    
    return results, success, z, measured_widths, analytical_widths, L_D


def benchmark_pure_spm():
    """Benchmark 4: Pure SPM (β₂=0) - temporal shape unchanged, spectrum broadens."""
    print("\n" + "="*60)
    print("Benchmark 4: Pure SPM (No Dispersion)")
    print("="*60)
    
    # Parameters
    wavelength_nm = 1550
    fwhm_ps = 0.1
    gamma_W_m = 1.0
    fiber_length = 0.01  # 1 cm
    
    # Create pulse with enough power for significant SPM
    pulse = lf.Pulse(
        pulse_type='gaussian',
        fwhm_ps=fwhm_ps,
        center_wavelength_nm=wavelength_nm,
        time_window_ps=10,
        npts=2**13,
        epp=100e-12,  # 100 pJ
    )
    
    P0 = np.max(pulse.it)
    
    # Fiber with no dispersion
    fiber = lf.Fiber(
        length=fiber_length,
        center_wl_nm=wavelength_nm,
        dispersion=[0],  # No dispersion
        gamma_W_m=gamma_W_m,
        loss_dB_per_m=0,
    )
    
    # Expected B-integral (max nonlinear phase)
    B_integral = gamma_W_m * P0 * fiber_length
    print(f"Peak power P₀ = {P0:.1f} W")
    print(f"Expected B-integral: {B_integral:.2f} rad")
    
    results = lf.NLSE(pulse, fiber, nsaves=50, print_status=False,
                      raman=False, shock=False, atol=1e-8, rtol=1e-8)
    
    z, f, t, AW, AT = results.get_results(data_type='intensity')
    
    # Check temporal shape is preserved
    initial_temporal = AT[0] / np.max(AT[0])
    final_temporal = AT[-1] / np.max(AT[-1])
    temporal_correlation = np.corrcoef(initial_temporal, final_temporal)[0, 1]
    
    # Check spectral broadening occurred
    initial_spectral = AW[0]
    final_spectral = AW[-1]
    
    # Spectral width (using second moment)
    f_center = np.sum(f * final_spectral) / np.sum(final_spectral)
    initial_sigma = np.sqrt(np.sum((f - f_center)**2 * initial_spectral) / np.sum(initial_spectral))
    final_sigma = np.sqrt(np.sum((f - f_center)**2 * final_spectral) / np.sum(final_spectral))
    spectral_broadening = final_sigma / initial_sigma
    
    print(f"\nResults:")
    print(f"Temporal correlation (should be ~1): {temporal_correlation:.4f}")
    print(f"Spectral broadening factor: {spectral_broadening:.2f}×")
    
    success = temporal_correlation > 0.99 and spectral_broadening > 1.1
    print(f"✓ PASSED" if success else "✗ FAILED")
    
    return results, success, temporal_correlation, spectral_broadening


def run_example():
    """Run all benchmarks and create summary plots."""
    print("="*60)
    print("NLSE Analytical Benchmarks")
    print("="*60)
    
    # Run all benchmarks and capture metrics
    res1, pass1, peak_var1 = benchmark_fundamental_soliton()
    res2, pass2, corr2, peak_ratio2 = benchmark_higher_order_soliton()
    res3, pass3, z3, meas3, ana3, LD3 = benchmark_linear_dispersion()
    res4, pass4, temp_corr4, spec_broad4 = benchmark_pure_spm()
    
    # Calculate additional metrics for plot 3
    max_err3 = np.max(np.abs(meas3 - ana3) / ana3) * 100
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = all([pass1, pass2, pass3, pass4])
    print(f"1. Fundamental soliton: {'✓' if pass1 else '✗'}")
    print(f"2. Higher-order soliton: {'✓' if pass2 else '✗'}")
    print(f"3. Linear dispersion: {'✓' if pass3 else '✗'}")
    print(f"4. Pure SPM: {'✓' if pass4 else '✗'}")
    print(f"\nOverall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    
    # Define two tolerances for comparison (factor of 5 apart)
    tol_coarse = 1e-6
    tol_fine = 1e-8  # This matches the hardcoded tolerance in benchmark functions
    
    # Re-run soliton and SPM benchmarks at coarse tolerance for comparison
    # (dispersion benchmark already uses 1e-10 and is at floating-point limit)
    print("\nRunning additional simulations for tolerance comparison...")
    
    # N=1 soliton at coarse tolerance
    wavelength_nm, T0_ps, beta2_ps2_m, gamma_W_m = 1550, 0.1, -0.020, 1.0
    pulse1c = create_soliton_pulse(N=1, T0_ps=T0_ps, wavelength_nm=wavelength_nm,
                                    beta2_ps2_m=beta2_ps2_m, gamma_W_m=gamma_W_m)
    fiber1c = lf.Fiber(length=0.05, center_wl_nm=wavelength_nm, dispersion=[beta2_ps2_m],
                       gamma_W_m=gamma_W_m, loss_dB_per_m=0)
    res1c = lf.NLSE(pulse1c, fiber1c, nsaves=50, print_status=False, raman=False, shock=False,
                    atol=tol_coarse, rtol=tol_coarse)
    
    # N=2 soliton at coarse tolerance
    T0_s = T0_ps * 1e-12
    beta2_s2_m = beta2_ps2_m * 1e-24
    L_D = T0_s**2 / np.abs(beta2_s2_m)
    z0 = (np.pi / 2) * L_D
    pulse2c = create_soliton_pulse(N=2, T0_ps=T0_ps, wavelength_nm=wavelength_nm,
                                    beta2_ps2_m=beta2_ps2_m, gamma_W_m=gamma_W_m)
    fiber2c = lf.Fiber(length=z0, center_wl_nm=wavelength_nm, dispersion=[beta2_ps2_m],
                       gamma_W_m=gamma_W_m, loss_dB_per_m=0)
    res2c = lf.NLSE(pulse2c, fiber2c, nsaves=100, print_status=False, raman=False, shock=False,
                    atol=tol_coarse, rtol=tol_coarse)
    
    # Pure SPM at coarse tolerance
    pulse4c = lf.Pulse(pulse_type='gaussian', fwhm_ps=0.1, center_wavelength_nm=1550,
                       time_window_ps=10, npts=2**13, epp=100e-12)
    fiber4c = lf.Fiber(length=0.01, center_wl_nm=1550, dispersion=[0], gamma_W_m=1.0, loss_dB_per_m=0)
    res4c = lf.NLSE(pulse4c, fiber4c, nsaves=50, print_status=False, raman=False, shock=False,
                    atol=tol_coarse, rtol=tol_coarse)
    
    # Linear dispersion comparison: grid resolution matters, not tolerance
    # (dispersion is applied analytically in Fourier space)
    fwhm_ps_disp = 0.1
    beta2_ps2_m_disp = -0.020
    T0_ps_disp = fwhm_ps_disp / (2 * np.sqrt(np.log(2)))
    T0_s_disp = T0_ps_disp * 1e-12
    beta2_s2_m_disp = beta2_ps2_m_disp * 1e-24
    L_D_disp = T0_s_disp**2 / np.abs(beta2_s2_m_disp)
    
    # Coarse grid (2^11 = 2048 points)
    npts_coarse = 2**11
    pulse3c = lf.Pulse(pulse_type='gaussian', fwhm_ps=fwhm_ps_disp, center_wavelength_nm=1550,
                       time_window_ps=20, npts=npts_coarse, epp=1e-12)
    fiber3c = lf.Fiber(length=5*L_D_disp, center_wl_nm=1550, dispersion=[beta2_ps2_m_disp],
                       gamma_W_m=0, loss_dB_per_m=0)
    res3c = lf.NLSE(pulse3c, fiber3c, nsaves=51, print_status=False, raman=False, shock=False,
                    atol=1e-8, rtol=1e-8)
    z3c, _, t3c, _, AT3c = res3c.get_results(data_type='intensity')
    
    # Measure widths using second moment (RMS)
    meas3c = []
    for i in range(len(z3c)):
        profile = AT3c[i]
        total_power = np.sum(profile)
        t_mean = np.sum(t3c * profile) / total_power
        sigma_rms = np.sqrt(np.sum((t3c - t_mean)**2 * profile) / total_power)
        width = sigma_rms * 2 * np.sqrt(2 * np.log(2))
        meas3c.append(width)
    meas3c = np.array(meas3c)
    
    # Create figure with 4x2 subplots (main plot + residuals for each benchmark)
    fig, axes = plt.subplots(4, 2, figsize=(12, 8), 
                             gridspec_kw={'height_ratios': [3, 1, 3, 1]})
    
    # Plot 1: N=1 soliton - peak power vs z
    z1, f1, t1, AW1, AT1 = res1.get_results(data_type='intensity')
    peaks1 = np.max(AT1, axis=1)
    norm_peaks1 = peaks1 / peaks1[0]
    theory1 = np.ones_like(norm_peaks1)
    
    z1c, _, _, _, AT1c = res1c.get_results(data_type='intensity')
    peaks1c = np.max(AT1c, axis=1)
    norm_peaks1c = peaks1c / peaks1c[0]
    
    axes[0, 0].plot(z1 * 1e3, norm_peaks1, 'b-', linewidth=2, label='Simulation')
    axes[0, 0].axhline(1.0, color='r', linestyle='--', label='Theory (constant)')
    axes[0, 0].set_ylabel('Normalized peak power')
    axes[0, 0].text(0.02, 0.98, f'(a) N=1 Soliton Stability (tol={tol_fine:.0e})', transform=axes[0, 0].transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_ylim(0.99, 1.01)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].text(0.95, 0.05, f'Peak variation: {peak_var1*100:.4f}%',
                    transform=axes[0, 0].transAxes, ha='right', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals for plot 1 - both tolerances
    residual1 = (norm_peaks1 - theory1) * 100  # percent (fine tol)
    residual1c = (norm_peaks1c - theory1) * 100  # percent (coarse tol)
    axes[1, 0].plot(z1c * 1e3, residual1c, 'C1', linewidth=1, alpha=0.7, label=f'tol={tol_coarse:.0e}')
    axes[1, 0].plot(z1 * 1e3, residual1, 'C0', linewidth=1, label=f'tol={tol_fine:.0e}')
    axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Propagation distance (mm)')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 2: N=2 soliton - input vs output profile
    z2, f2, t2, AW2, AT2 = res2.get_results(data_type='intensity')
    input2 = AT2[0] / np.max(AT2[0])
    output2 = AT2[-1] / np.max(AT2[-1])
    
    _, _, t2c, _, AT2c = res2c.get_results(data_type='intensity')
    input2c = AT2c[0] / np.max(AT2c[0])
    output2c = AT2c[-1] / np.max(AT2c[-1])
    
    axes[0, 1].plot(t2, input2, 'b-', linewidth=2, label='Input (z=0)')
    axes[0, 1].plot(t2, output2, 'r--', linewidth=2, label='Output (z=z₀)')
    axes[0, 1].set_ylabel('Normalized intensity')
    axes[0, 1].text(0.02, 0.98, f'(b) N=2 Soliton Period (tol={tol_fine:.0e})', transform=axes[0, 1].transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].text(0.95, 0.05, f'Correlation: {corr2:.6f}\nPeak ratio: {peak_ratio2:.6f}',
                    transform=axes[0, 1].transAxes, ha='right', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals for plot 2 - both tolerances
    residual2 = (output2 - input2) * 100  # percent difference (fine)
    residual2c = (output2c - input2c) * 100  # percent difference (coarse)
    axes[1, 1].plot(t2c, residual2c, 'C1', linewidth=1, alpha=0.7, label=f'tol={tol_coarse:.0e}')
    axes[1, 1].plot(t2, residual2, 'C0', linewidth=1, label=f'tol={tol_fine:.0e}')
    axes[1, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Time (ps)')
    axes[1, 1].set_ylabel('Diff (%)')
    axes[1, 1].set_xlim(-1, 1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 3: Linear dispersion - width vs z
    norm_meas3 = meas3 / meas3[0]
    norm_ana3 = ana3 / meas3[0]
    
    axes[2, 0].plot(z3 / LD3, norm_meas3, 'bo', markersize=4, label='Simulation')
    axes[2, 0].plot(z3 / LD3, norm_ana3, 'r-', linewidth=2, label='Analytical')
    axes[2, 0].set_ylabel('τ(z) / τ₀')
    axes[2, 0].text(0.02, 0.98, '(c) Linear Dispersion (npts=2¹³)', transform=axes[2, 0].transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold')
    axes[2, 0].legend(loc='upper right')
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].text(0.95, 0.05, f'Max error: {max_err3:.4f}%',
                    transform=axes[2, 0].transAxes, ha='right', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals for plot 3 - different grid resolutions (dispersion is grid-limited)
    norm_meas3c = meas3c / meas3c[0]
    z3c_norm = z3c / L_D_disp
    ana3c = np.sqrt(1 + z3c_norm**2) * meas3c[0]
    norm_ana3c = ana3c / meas3c[0]
    
    residual3 = (norm_meas3 - norm_ana3) / norm_ana3  # relative error
    residual3c = (norm_meas3c - norm_ana3c) / norm_ana3c  # (coarse grid)
    axes[3, 0].plot(z3c_norm, residual3c, 'C1', linewidth=1, alpha=0.7, label=f'npts=2¹¹')
    axes[3, 0].plot(z3 / LD3, residual3, 'C0', linewidth=1, label=f'npts=2¹³')
    axes[3, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[3, 0].set_xlabel('z / L_D')
    axes[3, 0].set_ylabel('Relative Error')
    axes[3, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(alpha=0.3)
    
    # Plot 4: Pure SPM - temporal and spectral
    z4, f4, t4, AW4, AT4 = res4.get_results(data_type='intensity')
    input4 = AT4[0] / np.max(AT4[0])
    output4 = AT4[-1] / np.max(AT4[-1])
    
    _, _, t4c, _, AT4c = res4c.get_results(data_type='intensity')
    input4c = AT4c[0] / np.max(AT4c[0])
    output4c = AT4c[-1] / np.max(AT4c[-1])
    
    axes[2, 1].plot(t4, input4, 'b-', linewidth=2, label='Temporal (z=0)')
    axes[2, 1].plot(t4, output4, 'r--', linewidth=2, alpha=0.7, label='Temporal (z=L)')
    axes[2, 1].set_ylabel('Normalized intensity')
    axes[2, 1].set_xlim(-0.5, 0.5)
    axes[2, 1].legend(loc='upper right')
    axes[2, 1].text(0.02, 0.98, f'(d) Pure SPM (tol={tol_fine:.0e})', transform=axes[2, 1].transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold')
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].text(0.95, 0.05, f'Temporal corr: {temp_corr4:.6f}\nSpectral broad: {spec_broad4:.2f}×',
                    transform=axes[2, 1].transAxes, ha='right', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals for plot 4 - both tolerances
    residual4 = (output4 - input4) * 100  # absolute difference in % (fine)
    residual4c = (output4c - input4c) * 100  # (coarse)
    axes[3, 1].plot(t4c, residual4c, 'C1', linewidth=1, alpha=0.7, label=f'tol={tol_coarse:.0e}')
    axes[3, 1].plot(t4, residual4, 'C0', linewidth=1, label=f'tol={tol_fine:.0e}')
    axes[3, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[3, 1].set_xlabel('Time (ps)')
    axes[3, 1].set_ylabel('Diff (%)')
    axes[3, 1].set_xlim(-0.5, 0.5)
    axes[3, 1].legend(fontsize=8)
    axes[3, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('NLSE_analytical_benchmarks.png', dpi=250)
    print("Saved plot to NLSE_analytical_benchmarks.png")
    plt.show()
    
    return all_passed


if __name__ == "__main__":
    run_example()
