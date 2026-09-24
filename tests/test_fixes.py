import warnings
import numpy as np
import pytest
from pyrsd.utils.io import image_to_hue_field
from pyrsd.core.calibration import fit_spline, find_saturation_plateau, estimate_noise
from pyrsd.core.processing import hue_to_displacement, plateau_mask
from pyrsd.core.solvers.integration import integrate_1d
from pyrsd.core.filters import gaussian_filter, median_filter


def _synthetic_calibration(seed=0, sigma_h=0.4):
    """monotone hue(d) with a plateau at 240 deg, noise on hue like a real ROI mean"""
    rng = np.random.default_rng(seed)
    d = np.arange(0.1, 3.4, 0.05)
    h = 60 + 75 * d - 6 * d**2
    h = np.where((d >= 2.6) & (d <= 2.95), 240.0, h + np.where(d > 2.95, 15, 0))
    h = h + rng.normal(0, sigma_h, d.size) * ~((d >= 2.6) & (d <= 2.95))
    return h, d


def test_dark_pixels_are_masked_but_bright_ones_kept():
    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = (0, 1, 0)      # BGR near-black green: S = 1, V = 1/255
    img[0, 1] = (0, 90, 0)     # dim green, V = 0.35
    img[1, 0] = (0, 10, 0)     # V = 0.04 < 0.05
    img[1, 1] = (0, 13, 0)     # V = 0.051
    hue = image_to_hue_field(img)
    assert np.isnan(hue[0, 0]) and np.isnan(hue[1, 0])
    assert hue[0, 1] == pytest.approx(120.0) and hue[1, 1] == pytest.approx(120.0)
    assert not np.isnan(image_to_hue_field(img, val_threshold=0.0)[0, 0])  # legacy behaviour


def test_plateau_detected_and_collapsed():
    h, d = _synthetic_calibration()
    m = find_saturation_plateau(h)
    true = np.isclose(h, 240.0)
    assert (m == true).all()
    with pytest.warns(RuntimeWarning):
        s = fit_spline(h, d, "spline")
    assert s.plateau_hue == pytest.approx(240.0)
    assert s.plateau_halfwidth_mm == pytest.approx(0.5 * (d[true].max() - d[true].min()))
    assert s(240.0) == pytest.approx(d[true].mean(), abs=0.05)


@pytest.mark.parametrize("fit", ["linear", "cubic"])
def test_interpolating_fits_no_longer_crash_on_plateau(fit):
    h, d = _synthetic_calibration()
    with pytest.warns(RuntimeWarning):
        s = fit_spline(h, d, fit)
    assert np.isfinite(s(120.0))


def test_interpolating_fit_with_plateau_kept_raises_clear_error():
    h, d = _synthetic_calibration()
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_spline(h, d, "cubic", plateau="keep")


def test_default_spline_follows_calibration_not_a_global_cubic():
    h, d = _synthetic_calibration(sigma_h=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = fit_spline(h, d, "spline")
        old = fit_spline(h, d, "spline", sigma_mm=1.0)   # == old scipy default s = N
    ok = ~find_saturation_plateau(h)
    rms_new = np.sqrt(np.mean((s(h[ok]) - d[ok]) ** 2))
    rms_old = np.sqrt(np.mean((old(h[ok]) - d[ok]) ** 2))
    assert rms_new < 0.03
    assert rms_old > 3 * rms_new


def test_noise_estimate_recovers_known_scatter():
    rng = np.random.default_rng(1)
    h = np.linspace(20, 230, 200)
    d = 0.01 * h + rng.normal(0, 0.02, h.size)
    assert estimate_noise(h, d) == pytest.approx(0.02, rel=0.25)


def test_plateau_mask_flags_pixels():
    h, d = _synthetic_calibration()
    with pytest.warns(RuntimeWarning):
        s = fit_spline(h, d)
    field = np.array([[100.0, 240.0, 240.3, np.nan]])
    assert plateau_mask(field, s).tolist() == [[False, True, True, False]]


def test_integrate_recovers_analytic_profile():
    x = np.linspace(0, 10, 401); dx = x[1] - x[0]
    f = np.exp(-((x - 5) ** 2) / 2)
    grad = np.gradient(f, dx)[None, :]
    out = integrate_1d(grad, dx, ref_value=f[0], axis=1, ref_side="start")[0]
    assert np.max(np.abs(out - f)) < 1e-3


def test_integration_does_not_bridge_long_gap():
    g = np.ones((1, 30)); g[0, 12:22] = np.nan              # 10-pixel gap
    out = integrate_1d(g, 1.0, 0.0, axis=1, ref_side="start")[0]
    assert np.isfinite(out[:12]).all() and np.isnan(out[12:]).all()
    out_end = integrate_1d(g, 1.0, 0.0, axis=1, ref_side="end")[0]
    assert np.isnan(out_end[:22]).all() and np.isfinite(out_end[22:]).all()
    legacy = integrate_1d(g, 1.0, 0.0, axis=1, ref_side="start", max_gap=None)[0]
    assert np.isfinite(legacy).all()                        # old behaviour still available
    short = np.ones((1, 30)); short[0, 10:13] = np.nan      # 3-pixel gap is bridged
    assert np.isfinite(integrate_1d(short, 1.0, 0.0, axis=1)[0]).all()


def test_bad_ref_side_and_missing_index_raise():
    g = np.ones((1, 10))
    with pytest.raises(ValueError):
        integrate_1d(g, 1.0, 0.0, axis=1, ref_side="middle")
    with pytest.raises(ValueError):
        integrate_1d(g, 1.0, 0.0, axis=1, ref_side="index")


def test_bidirectional_was_a_noop_and_now_warns():
    rng = np.random.default_rng(2); g = rng.normal(size=(3, 40))
    a = integrate_1d(g, 0.5, 1.0, axis=1, ref_side="start")
    with pytest.warns(DeprecationWarning):
        b = integrate_1d(g, 0.5, 1.0, axis=1, ref_side="start", bidirectional=True)
    assert np.allclose(a, b)


def test_filters_do_not_bleed_zeros_from_masked_region():
    f = np.full((30, 30), 5.0); f[:, 20:] = np.nan
    g = gaussian_filter(f, 2.0); m = median_filter(f, 5)
    assert np.allclose(g[~np.isnan(f)], 5.0) and np.allclose(m[~np.isnan(f)], 5.0)
    assert np.isnan(g[:, 20:]).all() and np.isnan(m[:, 20:]).all()
