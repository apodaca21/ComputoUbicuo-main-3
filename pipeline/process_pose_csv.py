#!/usr/bin/env python3
"""
US/11 — Pipeline de procesamiento y normalización en coordenadas.

Uso:
    python process_pose_csv.py data/sample_pose_recording.csv
    python process_pose_csv.py ruta/a/grabacion.csv --output-dir output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from features import compute_upper_body_distances, distances_to_tensor
from loader import load_pose_csv
from minmax_scaling import fit_minmax, transform_minmax, transform_tensor_minmax
from spatial_filter import apply_spatial_subset, spatial_filter_report


def run_pipeline(
    csv_path: Path,
    output_dir: Path,
    min_confidence: float = 0.2,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_pose_csv(csv_path)
    filtered = apply_spatial_subset(raw, min_confidence=min_confidence)
    report = spatial_filter_report(raw, filtered)

    distances = compute_upper_body_distances(filtered)
    scaler = fit_minmax(distances)
    distances_scaled = transform_minmax(distances, scaler)

    tensor_raw = distances_to_tensor(distances)
    tensor_scaled = transform_tensor_minmax(tensor_raw, scaler)

    # Artefactos
    filtered_path = output_dir / "01_spatial_filtered.csv"
    distances_path = output_dir / "02_distances_raw.csv"
    scaled_path = output_dir / "03_distances_minmax.csv"
    tensor_raw_path = output_dir / "tensor_raw.npy"
    tensor_path = output_dir / "tensor_input.npy"
    scaler_path = output_dir / "minmax_scaler.json"
    meta_path = output_dir / "tensor_metadata.json"

    filtered.to_csv(filtered_path, index=False)
    distances.to_csv(distances_path, index=False)
    distances_scaled.to_csv(scaled_path, index=False)
    np.save(tensor_raw_path, tensor_raw)
    np.save(tensor_path, tensor_scaled)
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")

    metadata = {
        "source_csv": str(csv_path.resolve()),
        "spatial_filter": report,
        "tensor_shape": list(tensor_scaled.shape),
        "tensor_dtype": str(tensor_scaled.dtype),
        "axis_semantics": {
            "0": "sequence (track_id)",
            "1": "time (frame_id ordenado)",
            "2": "feature (distancia normalizada entre par de articulaciones)",
        },
        "feature_names": list(scaler.feature_names),
        "value_range": "[0, 1] tras Min-Max",
        "nan_policy": "NaN si falta articulación en el par; no se rellena en el tensor",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "filtered_path": filtered_path,
        "distances_path": distances_path,
        "scaled_path": scaled_path,
        "tensor_path": tensor_path,
        "tensor_shape": tensor_scaled.shape,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline US/11: CSV → filtrado espacial → distancias → Min-Max → tensor"
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Ruta al archivo .CSV con datos de pose grabados",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Directorio de salida (default: pipeline/output)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Umbral mínimo de confianza por articulación",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(args.csv_file, args.output_dir, args.min_confidence)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Pipeline US/11 completado.")
    print(f"  Filas tras filtrado espacial: {result['report']['rows_after']}")
    print(f"  Articulaciones eliminadas: {result['report']['joints_removed']}")
    print(f"  Tensor de entrada: {result['tensor_path']} shape={result['tensor_shape']}")
    print(f"  Salidas en: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
