"""Spectral grid of the IOWAGA hindcast point spectra used as Vetra targets."""

import numpy as np

# 36 frequencies, geometric with ratio 1.1 from 0.0339 to 0.9527 Hz.
IOWAGA_FREQUENCIES = np.array(
    [
        0.0339,
        0.03729,
        0.041019,
        0.0451209,
        0.04963299,
        0.05459629,
        0.06005592,
        0.06606152,
        0.07266767,
        0.07993444,
        0.08792789,
        0.09672067,
        0.10639274,
        0.11703202,
        0.12873523,
        0.14160876,
        0.15576965,
        0.17134662,
        0.18848129,
        0.20732942,
        0.22806236,
        0.25086862,
        0.27595547,
        0.30355102,
        0.33390614,
        0.3672968,
        0.40402648,
        0.44442913,
        0.48887205,
        0.5377593,
        0.5915352,
        0.6506888,
        0.7157577,
        0.78733355,
        0.866067,
        0.9526737,
    ],
    dtype=np.float64,
)
# 24 directions every 15 degrees, as the direction toward which waves travel.
IOWAGA_DIRECTIONS_TO = np.mod(90.0 - 15.0 * np.arange(24), 360.0).astype(np.float64)
