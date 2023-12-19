import functools

import numpy as np
import pandas as pd
from scipy import integrate, stats
from tlab_analysis import trpl


def make_trpl_data(
    lambda0: float = 230,
    sigma: float = 5,
    xi: float = 5e-2,
    tau: float = 0.2,
    seed: int = 0,
) -> trpl.TRPLData:
    def pulse(t: float, t0: float, sigma: float) -> float:
        return np.exp(-(((t - t0) / sigma) ** 2))  # type: ignore[no-any-return]

    random = np.random.RandomState(seed)
    time = np.linspace(0, 1, 480)
    wavelength = np.linspace(200, 300, 640)
    wv, tv = np.meshgrid(wavelength, time)
    input_fn = functools.partial(pulse, t0=0.2, sigma=1e-2)
    sol = integrate.solve_ivp(
        lambda t, y: -y / tau + input_fn(t),
        t_span=(time.min(), time.max()),
        y0=[0.0],
        method="Radau",
        t_eval=time,
    )
    intensity = (
        stats.exponnorm.pdf(
            wv,
            1 / (xi * sigma),
            loc=lambda0,
            scale=sigma,
        )
        * np.transpose(sol.y)
    ).astype(np.float64)
    max_intensity = float(intensity.max())
    intensity += random.normal(0, 1.0, intensity.shape) * max_intensity * 0.2
    intensity = (intensity / max_intensity * 10).astype(np.int64)
    intensity[intensity < 0] = 0
    return trpl.TRPLData(
        df=pd.DataFrame(
            dict(
                time=np.repeat(time, len(wavelength)),
                wavelength=np.tile(wavelength, len(time)),
                intensity=intensity.flatten(),
            )
        )
    )
