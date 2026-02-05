
"""
Benchmark Script: Integrator Comparison
Compares execution time vs. accuracy for different SciPy integrators.
Running this script will generate a plot 'integrator_benchmark.png'.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import laserfun as lf
import pandas as pd

def run_comparison():
    print("--- Laserfun Integrator Benchmark ---")
    print("Simulating Dudley et al. (2006) Supercontinuum Generation")

    # 1. Setup Simulation Parameters
    # We use a slightly reduced resolution to keep benchmark time reasonable,
    # but high enough to see differences.
    p = lf.Pulse(
        pulse_type="sech",
        power=10000,
        npts=2**11,  # 2048 points
        fwhm_ps=0.0284 * 1.76,
        center_wavelength_nm=835,
        time_window_ps=12.5,
    )

    betas = [
        -11.830e-3, 8.1038e-5, -9.5205e-8, 2.0737e-10,
        -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21, -1.7140e-24,
    ]
    f = lf.Fiber(length=0.15, center_wl_nm=835, dispersion=betas, gamma_W_m=0.11)

    # 2. Generate Ground Truth (High Precision)
    print("\nGenerating Ground Truth (lsoda, tol=1e-12)...")
    t0 = time.time()
    res_ref = lf.NLSE(p, f, nsaves=50, raman=True, custom_raman="dudley",
                      rtol=1e-12, atol=1e-12, print_status=True, 
                      integrator='lsoda', max_steps=1000000) # Reduced nsteps for sanity
    print(f"Ground Truth done in {time.time() - t0:.2f}s")
    
    # We use the final complex field (time domain) for error comparison
    # Error metric: L2 norm of difference / L2 norm of reference
    # Relative Error = ||E_test - E_ref|| / ||E_ref||
    ref_field = res_ref.AT[-1]
    ref_norm = np.linalg.norm(ref_field)

    # 3. Benchmark Loop
    integrators = ['lsoda', 'vode', 'dopri5', 'dop853']
    tolerances = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    
    results = []

    print("\nStarting comparison loop...")
    print(f"{'Integrator':<10} {'Tol':<10} {'Time':<10} {'Error'}")
    
    for name in integrators:
        # Default vode only (no custom method support)
        
        for tol in tolerances:
            # Clone pulse to specific instance
            p_run = p.create_cloned_pulse()
            
            print(f"Running {name} tol={tol}...", end=' ', flush=True)
            start_time = time.time()
            try:
                res = lf.NLSE(p_run, f, nsaves=50, raman=True, custom_raman="dudley",
                              rtol=tol, atol=tol, print_status=False, 
                              integrator=name, max_steps=1000000)
                elapsed = time.time() - start_time
                
                # Calculate Error
                test_field = res.AT[-1]
                diff_norm = np.linalg.norm(test_field - ref_field)
                rel_error = diff_norm / ref_norm
                
                print(f"{name:<10} {tol:<10.1e} {elapsed:<10.3f} {rel_error:.2e}")
                
                results.append({
                    'Integrator': name,
                    'Tolerance': tol,
                    'Time': elapsed,
                    'Error': rel_error,
                    'Label': f"{name} ({tol:.1e})"
                })
                
            except Exception as e:
                print(f"{name:<10} {tol:<10.1e} FAILED")

    # 4. Plotting
    print("\nPlotting results...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot each integrator as a separate line
    for name in integrators:
        subset = [r for r in results if r['Integrator'] == name]
        subset.sort(key=lambda x: x['Tolerance'], reverse=True) # Plot 1e-3 -> 1e-6
        
        times = [r['Time'] for r in subset]
        errors = [r['Error'] for r in subset]
        
        plt.loglog(times, errors, 'o-', label=name, linewidth=2, markersize=8)
        
        # Annotate points with tolerance? (optional, maybe too crowded)
        # for r in subset:
        #     plt.text(r['Time'], r['Error'], f"{r['Tolerance']:.0e}", fontsize=8)

    plt.xlabel("Execution Time (s)")
    plt.ylabel("Relative Error (vs. 1e-12 Reference)")
    plt.title("Laserfun Integrator Benchmark: Time vs. Accuracy\n(Bottom Left is Better)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    # plt.gca().invert_xaxis() # Removed inversion

    plt.tight_layout()
    plt.savefig('examples/benchmarks/integrator_benchmark.png', dpi=150)
    print("Saved plot to 'examples/benchmarks/integrator_benchmark.png'")
    plt.show()
if __name__ == "__main__":
    run_comparison()
