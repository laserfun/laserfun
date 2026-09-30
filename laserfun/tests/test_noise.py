"""Checks for Pulse.add_noise photon bookkeeping, PulseData.calc_coherence,
and the integrator step-budget error."""
import numpy as np
import pytest
import laserfun as lf
from laserfun.pulse import Pulse, h_planck


def _pulse():
    return Pulse(pulse_type="sech", fwhm_ps=0.2, time_window_ps=20, epp=100e-12, npts=2**12)


def test_epp_roundtrip():
    p = _pulse()
    assert np.isclose(p.epp, 100e-12)


def test_photons_per_bin_sums_to_pulse_photons():
    p = _pulse()
    n_total = p.photons_per_bin().sum()
    expected = p.epp / (h_planck * p.centerfrequency_THz * 1e12)
    assert np.isclose(n_total, expected, rtol=1e-3)


def test_add_noise_mean_photons_per_bin():
    rng = np.random.default_rng(0)
    for noise_type, expected in [("one_photon_freq", 1.0), ("half_photon_freq", 0.5)]:
        p = _pulse()
        aw0 = p.aw.copy()
        p.add_noise(noise_type, rng=rng)
        f_Hz = p.f_THz * 1e12
        df_Hz = p.df_THz * 1e12
        added = np.abs(p.aw - aw0) ** 2 / (df_Hz * h_planck * np.abs(f_Hz))
        assert np.isclose(added[f_Hz > 0].mean(), expected, rtol=0.05)
        assert np.all(added[f_Hz <= 0] == 0)


def test_add_noise_sqrt_N():
    rng = np.random.default_rng(2)
    p = _pulse()
    aw0 = p.aw.copy()
    N = p.photons_per_bin()
    f_Hz = p.f_THz * 1e12
    df_Hz = p.df_THz * 1e12
    # average the added photons over many realizations, since the sum is
    # dominated by the few bright bins near the spectral peak
    added = np.zeros_like(N)
    n_real = 50
    for _ in range(n_real):
        q = p.create_cloned_pulse()
        q.add_noise("sqrt_N_freq", rng=rng)
        added += np.abs(q.aw - aw0) ** 2 / (df_Hz * h_planck * np.abs(f_Hz))
    added /= n_real
    assert np.isclose(added.sum(), np.sqrt(N).sum(), rtol=0.1)


def test_add_noise_no_negative_frequencies():
    # a wide, dense grid extends to negative absolute frequencies
    p = Pulse(pulse_type="sech", fwhm_ps=0.01, time_window_ps=2, epp=1e-12, npts=2**12)
    f_Hz = p.f_THz * 1e12
    assert np.any(f_Hz <= 0)
    aw0 = p.aw.copy()
    p.add_noise("one_photon_freq", rng=np.random.default_rng(3))
    assert np.all(p.aw[f_Hz <= 0] == aw0[f_Hz <= 0])
    assert np.all(p.photons_per_bin()[f_Hz <= 0] == 0)


def test_add_noise_reproducible_and_rejects_unknown():
    p1, p2 = _pulse(), _pulse()
    p1.add_noise("one_photon_freq", rng=np.random.default_rng(5))
    p2.add_noise("one_photon_freq", rng=np.random.default_rng(5))
    assert np.array_equal(p1.aw, p2.aw)
    with pytest.raises(ValueError):
        _pulse().add_noise("not_a_noise_type")


def test_half_photon_gives_shot_noise_energy_variance():
    rng = np.random.default_rng(1)
    p = _pulse()
    N = p.photons_per_bin().sum()
    E = []
    for _ in range(300):
        q = p.create_cloned_pulse()
        q.add_noise("half_photon_freq", rng=rng)
        E.append(q.epp)
    E = np.array(E)
    assert np.isclose(E.std() / E.mean(), 1 / np.sqrt(N), rtol=0.15)


def _short_propagation(**kwargs):
    # bright enough (~1e9 photons) that one photon per bin of noise leaves
    # |g12| > 0.99 everywhere above 1e-3 of the spectral peak
    pulse = Pulse(pulse_type="sech", center_wavelength_nm=1550, fwhm_ps=0.1,
                  time_window_ps=10, epp=1e-10, npts=2**10)
    fiber = lf.Fiber(length=0.01, center_wl_nm=1550, dispersion_format="GVD",
                     dispersion=[-20e-3], gamma_W_m=1e-3, loss_dB_per_m=0)
    return pulse, fiber, lf.NLSE(pulse, fiber, nsaves=3, print_status=False, **kwargs)


def test_calc_coherence_runs_and_is_reproducible():
    pulse, fiber, results = _short_propagation()
    g12, trials = results.calc_coherence(num_trials=3, random_seed=0, nsaves=3)
    assert g12.shape == (3, pulse.npts)
    assert len(trials) == 3
    assert np.all(g12 >= 0) and np.all(g12 <= 1 + 1e-9)
    # a linear, low-power propagation of a bright pulse stays coherent
    f_THz = results.f_THz
    bright = np.abs(results.AW[-1]) ** 2 > 1e-3 * np.max(np.abs(results.AW[-1]) ** 2)
    assert np.all(g12[-1][bright] > 0.99)
    assert np.all(f_THz.shape == g12[-1].shape)

    g12b, _ = results.calc_coherence(num_trials=3, random_seed=0, nsaves=3)
    assert np.array_equal(g12, g12b)
    g12c, _ = results.calc_coherence(num_trials=3, random_seed=1, nsaves=3)
    assert not np.array_equal(g12, g12c)


def test_integrator_step_budget_raises():
    # exhausting max_steps on the *last* save interval used to return a
    # silently truncated propagation; it must now raise.
    pulse = Pulse(pulse_type="sech", center_wavelength_nm=835, fwhm_ps=0.05,
                  time_window_ps=12.5, epp=1e-9, npts=2**12)
    fiber = lf.Fiber(length=0.15, center_wl_nm=835, dispersion_format="GVD",
                     dispersion=[-11.830e-3, 8.1038e-5, -9.5205e-8],
                     gamma_W_m=0.11, loss_dB_per_m=0)
    with pytest.raises(RuntimeError, match="max_steps"):
        with pytest.warns(UserWarning):
            lf.NLSE(pulse, fiber, nsaves=2, integrator="dopri5", max_steps=20,
                    print_status=False, raman=True)
