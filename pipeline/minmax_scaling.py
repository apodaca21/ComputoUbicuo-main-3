"""Min-Max Scaling de las distancias del tren superior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DISTANCE_FEATURE_NAMES


@dataclass
class MinMaxScalerState:
    """Parámetros aprendidos para aplicar la misma escala en inferencia."""

    mins: np.ndarray  # shape (n_features,)
    maxs: np.ndarray  # shape (n_features,)
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
        }


def fit_minmax(distances_df: pd.DataFrame) -> MinMaxScalerState:
    """Calcula min y max por feature ignorando NaN."""
    cols = list(DISTANCE_FEATURE_NAMES)
    data = distances_df[cols].to_numpy(dtype=np.float64)
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    return MinMaxScalerState(
        mins=mins,
        maxs=maxs,
        feature_names=DISTANCE_FEATURE_NAMES,
    )


def transform_minmax(
    distances_df: pd.DataFrame,
    state: MinMaxScalerState,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Escala cada distancia a [0, 1]:
        x_scaled = (x - min) / (max - min + epsilon)
    """
    out = distances_df.copy()
    cols = list(state.feature_names)
    data = out[cols].to_numpy(dtype=np.float64)
    span = state.maxs - state.mins
    span = np.where(span < epsilon, epsilon, span)
    scaled = (data - state.mins) / span
    scaled = np.clip(scaled, 0.0, 1.0)
    out[cols] = scaled
    return out


def transform_tensor_minmax(
    tensor: np.ndarray,
    state: MinMaxScalerState,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Aplica Min-Max al tensor (n_seq, n_frames, n_features)."""
    span = state.maxs - state.mins
    span = np.where(span < epsilon, epsilon, span)
    scaled = (tensor - state.mins) / span
    return np.clip(scaled, 0.0, 1.0)
