# Vetra

Vetra estimates the directional wave spectrum at a fixed ocean point from the
history of the surface winds. Two 3D U-Net encoders read the winds over an
ocean basin (10 days on a 2° grid) and around the point (3 days on a 0.5°
grid), and a convolutional decoder turns their features into the spectrum on
36 frequencies × 24 directions. Vetra is trained on hindcast spectra, one
model per site, and has an optional post-correction, Hs-Gain, that rescales
the energy of each spectrum according to its significant wave height.

Nothing in the code is tied to a particular site: any point with wind fields
and spectra in the format below can be used.

The code is archived on Zenodo
([doi:10.5281/zenodo.24400745](https://doi.org/10.5281/zenodo.24400745)).
Versions up to 1.0.0 were released under the name Njord.

## Installation

```bash
pip install -e .
```

Requires Python 3.10 or later, PyTorch 2 and NumPy. A GPU is needed for
training; inference runs on a CPU.

## Pretrained models

Trained models for four sites are available on Zenodo
([doi:10.5281/zenodo.23303893](https://doi.org/10.5281/zenodo.23303893)):
West Bermuda (NDBC 41048), East Gulf (42003), Southwest Hawaii (51002) and
Stratus (32012). They were trained on 1993 to 2021 and each one is valid only
for its site. Each file contains the weights, the wind normalisation, the
Hs-Gain table and a description of the input domains:

```python
from vetra import load_checkpoint

vetra = load_checkpoint("vetra_southwest_hawaii.pt", device="cuda")
print(vetra.site)                              # point, buoy, wind domains
spectra = vetra.predict(large, local)          # (N, 36, 24), Hs-Gain applied
raw = vetra.predict(large, local, apply_hs_gain=False)
```

`large` has shape (N, 80, 2, H, W) and `local` has shape (N, 24, 2, h, w):
for each of the N spectra, the last 80 and the last 24 three-hourly wind
fields up to and including the time of the spectrum, prepared as described
below on the domains given in `vetra.site`.

## Using Vetra at a new site

### 1. Prepare the data

Write one file per month under a folder named after the site:

```
data/<site>/<YYYY>/large_<YYYYMM>.npy      (T, 2, H, W)   basin winds
data/<site>/<YYYY>/local_<YYYYMM>.npy      (T, 2, h, w)   local winds
data/<site>/<YYYY>/spectrum_<YYYYMM>.npy   (T, 36, 24)    spectra
```

The three files of a month must share the same T time steps, every 3 h
(00, 03, ..., 21 UTC), and every year needs all twelve months.

- **Winds**: ERA5 10-m wind components (`10m_u_component_of_wind`,
  `10m_v_component_of_wind`, Copernicus Climate Data Store), stored as their
  signed squares, channel 0 = u|u| and channel 1 = v|v|, in float32.
  Basin and local grids can have any size of at least 16 × 16 points. The
  published models use a 2° grid over the ocean basin that sends swell to the
  site, and a 0.5° grid over the site ± 20° (81 × 81).
- **Spectra**: energy density E(f, θ) in m² s rad⁻¹ on the grid of
  `vetra/grid.py`: the 36 frequencies (0.0339 to 0.9527 Hz) and 24
  directions of the IOWAGA WAVEWATCH III hindcast point output (Ifremer).
  Spectra from another source must be interpolated onto this grid.

For example, from monthly ERA5 NetCDF files:

```python
import numpy as np
import xarray as xr

def signed_square_winds(path):
    ds = xr.open_dataset(path)
    u, v = ds["u10"].values, ds["v10"].values
    return np.stack([u * np.abs(u), v * np.abs(v)], axis=1).astype(np.float32)

np.save("data/MySite/2020/large_202001.npy", signed_square_winds("era5_basin_202001.nc"))
np.save("data/MySite/2020/local_202001.npy", signed_square_winds("era5_local_202001.nc"))
```

### 2. Train

```bash
python scripts/train.py --site MySite --year $(seq 1993 2021) --data-root data --out-dir runs/mysite
```

The defaults are the recipe of the paper: MSE on log(1 + E), Lion optimiser
(learning rate 1e-4), batch size 16, at most 70 epochs, and a chronological
80/20 split of the training years into training and validation data. The
checkpoint is the epoch with the highest physical correlation on the
validation data. The learning rate follows RR-4 (restore and reduce): after
four epochs without improvement, the best weights are restored and the
learning rate is multiplied by 0.2, at most four times. At the end, Hs-Gain
is fitted on the validation data. The model, the normalisation and the
Hs-Gain table are written to `runs/mysite/vetra.pt`, which `load_checkpoint`
reads as above. An interrupted run continues with `--resume`.

The published models were trained on 29 years; with fewer years the model
sees fewer storms and Hs-Gain rests on fewer large waves.

## Tests

```bash
pip install pytest
pytest tests
```

## Citation

If you use Vetra, please cite:

> Farina, L. Can Winds Alone Predict Ocean Wave Spectra? Submitted.

## License

MIT. See `LICENSE`.
