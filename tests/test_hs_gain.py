import numpy as np

from vetra.hs_gain import HsGain, fit_hs_gain, significant_wave_height


def test_fit_recovers_constant_gain_and_keeps_shape(tmp_path):
    rng = np.random.default_rng(0)
    target = rng.gamma(2.0, 0.05, size=(500, 36, 24))
    prediction = target / 1.3          # energy underestimated by a factor 1.3 everywhere
    gain = fit_hs_gain(prediction, target)
    np.testing.assert_allclose(gain.energy_gain, 1.3, rtol=1e-6)

    gain.save(tmp_path / "hs_gain.json")
    corrected = HsGain.load(tmp_path / "hs_gain.json").apply(prediction)
    np.testing.assert_allclose(corrected, target, rtol=1e-5)
    np.testing.assert_allclose(significant_wave_height(corrected), significant_wave_height(target), rtol=1e-5)


def test_gain_is_monotone_and_constant_beyond_fitted_range():
    gain = HsGain(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.1, 1.2]))
    spectrum = np.ones((1, 36, 24))
    hs = significant_wave_height(spectrum)[0]
    big = spectrum * (10.0 / hs) ** 2   # Hs = 10 m, beyond the last centre
    np.testing.assert_allclose(gain.apply(big) / big, 1.2)

    rng = np.random.default_rng(1)
    prediction = rng.gamma(2.0, 0.05, size=(400, 36, 24))
    target = prediction * rng.uniform(0.5, 1.5, size=(400, 1, 1))
    assert np.all(np.diff(fit_hs_gain(prediction, target).energy_gain) >= 0)


def test_isotonic_pools_violators_instead_of_carrying_the_maximum():
    from vetra.hs_gain import isotonic_increasing
    fitted = isotonic_increasing(np.array([1.3, 1.0, 1.1, 1.2]), np.ones(4))
    np.testing.assert_allclose(fitted, [3.4 / 3, 3.4 / 3, 3.4 / 3, 1.2])
