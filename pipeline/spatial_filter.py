"""Filtrado espacial (Subset Selection): elimina articulaciones irrelevantes."""

from __future__ import annotations

import pandas as pd

from config import EXCLUDED_JOINTS, UPPER_BODY_JOINTS


def apply_spatial_subset(df: pd.DataFrame, min_confidence: float = 0.2) -> pd.DataFrame:
    """
    Elimina rostro, rodillas y tobillos; conserva solo el tren superior.

    Parámetros
    ----------
    df : DataFrame con columnas joint, x, y, confidence, ...
    min_confidence : umbral mínimo para descartar mediciones ruidosas.

    Retorna
    -------
    DataFrame filtrado solo con articulaciones de UPPER_BODY_JOINTS.
    """
    out = df[~df["joint"].isin(EXCLUDED_JOINTS)].copy()
    out = out[out["joint"].isin(UPPER_BODY_JOINTS)]
    out = out[out["confidence"] >= min_confidence]
    return out.reset_index(drop=True)


def spatial_filter_report(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Resumen del filtrado para logs."""
    removed = set(before["joint"].unique()) - set(after["joint"].unique())
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "joints_removed": sorted(removed),
        "joints_kept": sorted(after["joint"].unique()),
    }
