
import numpy as np
import matplotlib.pyplot as plt
import laserfun as lf
from laserfun import tools

def compare_phases():
    # Parameters
    wl = 1545.0 # nm
    lines = 1000 # l/mm
    separation = 1e-2 # 10 mm
    fwhm_ps = 0.1 # 100 fs
    
    # 1. Initialize Compressor (Littrow)
    comp = tools.TreacyCompressor(lines_per_mm=lines, littrow_wavelength_nm=wl)
    
    # 2. Setup Pulses
    # Need sufficient potential window for the chirped pulse
    npts = 2**14
    time_window = 40.0 # ps
    
    # Pulse 1: Full Compressor Phase
    p1 = lf.Pulse(pulse_type='gaussian', fwhm_ps=fwhm_ps, center_wavelength_nm=wl,
                  time_window_ps=time_window, npts=npts)
    
    # Apply Compressor Phase to p1
    comp.apply_phase_to_pulse(separation, p1)
    
    # Apply Manual GDD/TOD
    gdd = comp.calc_compressor_gdd(wl, separation)
    tod = comp.calc_compressor_HOD(wl, separation, 3)
    
    print(f"Applied GDD: {gdd*1e24:.4f} ps^2")
    print(f"Applied TOD: {tod*1e36:.4f} ps^3")
    
    # Pulse 2: Create with GDD and TOD directly
    # Note: Pulse expects GDD in ps^2 and TOD in ps^3 (uppercase kwargs)
    p2 = lf.Pulse(pulse_type='gaussian', fwhm_ps=fwhm_ps, center_wavelength_nm=wl,
                  time_window_ps=time_window, npts=npts,
                  GDD=gdd*1e24, TOD=tod*1e36)
    
    # Plotting
    t_ps = p1.t_ps
    
    # Check peak positions
    peak1 = t_ps[np.argmax(np.abs(p1.at)**2)]
    peak2 = t_ps[np.argmax(np.abs(p2.at)**2)]
    print(f"Peak 1 position: {peak1:.6f} ps")
    print(f"Peak 2 position: {peak2:.6f} ps")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_ps, np.abs(p1.at)**2, 'r-', linewidth=2, label='Full Compressor (Treacy)')
    plt.plot(t_ps, np.abs(p2.at)**2, 'b--', linewidth=2, label='GDD + TOD Approx')
    
    plt.xlabel('Time (ps)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Comparison @ 1 mm separation\nGDD={gdd*1e24:.3f} ps$^2$, TOD={tod*1e36:.5f} ps$^3$')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-5, 5) # Zoom in on the pulse
    plt.tight_layout()
    plt.savefig('compressor_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    compare_phases()
