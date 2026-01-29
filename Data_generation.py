import numpy as np
import pandas as pd

def generate_data(T=2500, features=5, seed=42):
    np.random.seed(seed)
    t = np.arange(T)

    season1 = np.sin(2 * np.pi * t / 50)
    season2 = np.sin(2 * np.pi * t / 200)
    trend = 0.0005 * t**1.2

    base = np.vstack([
        season1 + trend,
        season2 + 0.5*trend,
        0.7*season1 + 0.3*season2,
        np.cos(2*np.pi*t/100),
        np.sin(2*np.pi*t/150)
    ]).T

    corr = np.array([
        [1.0,0.8,0.6,0.3,0.2],
        [0.8,1.0,0.5,0.2,0.1],
        [0.6,0.5,1.0,0.4,0.3],
        [0.3,0.2,0.4,1.0,0.5],
        [0.2,0.1,0.3,0.5,1.0]
    ])

    L = np.linalg.cholesky(corr)
    correlated = base @ L.T

    volatility = 0.2 + 0.3*np.sin(2*np.pi*t/300)
    noise = np.random.randn(T, features) * volatility[:, None]

    data = correlated + noise

    # Missing values
    mask = np.random.rand(*data.shape) < 0.05
    data[mask] = np.nan

    df = pd.DataFrame(data).interpolate().bfill().ffill()
    return df
