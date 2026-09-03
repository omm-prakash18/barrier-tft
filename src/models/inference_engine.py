"""
inference_engine.py
───────────────────
High-Throughput, Low-Latency Inference Engine for Cross-Sectional TFT.

Executes quantitative predictions via ONNX Runtime / TorchScript with:
  - Multi-threaded CPU / CUDA / TensorRT execution provider routing.
  - Zero-overhead batch inference.
  - Microsecond latency benchmarking (p50, p95, p99 metrics).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch


class TFTInferenceEngine:
    """
    High-Performance Inference Engine for Production Deployment.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        execution_providers: Optional[List[str]] = None,
        num_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.backend = "onnx" if self.model_path.suffix == ".onnx" else "torchscript"
        self._session = None
        self._ts_model = None

        if self.backend == "onnx":
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = num_threads
                opts.inter_op_num_threads = num_threads
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                providers = execution_providers or ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(self.model_path), sess_options=opts, providers=providers
                )
            except Exception:
                # Fallback to TorchScript companion if onnxruntime is not installed
                self.backend = "torchscript"

        if self.backend == "torchscript":
            ts_file = self.model_path if self.model_path.suffix == ".pt" else self.model_path.with_suffix(".pt")
            if ts_file.exists():
                self._ts_model = torch.jit.load(str(ts_file), map_location="cpu")
                self._ts_model.eval()

    def predict(
        self,
        ticker_ids: np.ndarray,
        sector_ids: np.ndarray,
        regime_ids: np.ndarray,
        x_enc: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Execute sub-millisecond batch inference.
        """
        if self.backend == "onnx" and self._session is not None:
            inputs = {
                "ticker_ids": ticker_ids.astype(np.int64),
                "sector_ids": sector_ids.astype(np.int64),
                "regime_ids": regime_ids.astype(np.int64),
                "x_enc": x_enc.astype(np.float32),
            }
            p10, p50, p90 = self._session.run(None, inputs)
            return {
                "p10": np.asarray(p10).reshape(-1),
                "p50": np.asarray(p50).reshape(-1),
                "p90": np.asarray(p90).reshape(-1),
            }

        elif self._ts_model is not None:
            with torch.no_grad():
                t_ids = torch.from_numpy(ticker_ids.astype(np.int64))
                s_ids = torch.from_numpy(sector_ids.astype(np.int64))
                r_ids = torch.from_numpy(regime_ids.astype(np.int64))
                x_e = torch.from_numpy(x_enc.astype(np.float32))
                p10, p50, p90 = self._ts_model(t_ids, s_ids, r_ids, x_e)
                return {
                    "p10": p10.cpu().numpy().reshape(-1),
                    "p50": p50.cpu().numpy().reshape(-1),
                    "p90": p90.cpu().numpy().reshape(-1),
                }
        else:
            raise RuntimeError(f"No valid inference runtime loaded from {self.model_path}")

    def benchmark_latency(
        self,
        n_iterations: int = 100,
        batch_size: int = 32,
        seq_len: int = 20,
        n_features: int = 6,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency and return p50, p95, p99 metrics in microseconds.
        """
        ticker_ids = np.zeros(batch_size, dtype=np.int64)
        sector_ids = np.zeros(batch_size, dtype=np.int64)
        regime_ids = np.zeros(batch_size, dtype=np.int64)
        x_enc = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(ticker_ids, sector_ids, regime_ids, x_enc)

        latencies_us: List[float] = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.predict(ticker_ids, sector_ids, regime_ids, x_enc)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000.0)

        return {
            "p50_us": float(np.percentile(latencies_us, 50)),
            "p95_us": float(np.percentile(latencies_us, 95)),
            "p99_us": float(np.percentile(latencies_us, 99)),
            "mean_us": float(np.mean(latencies_us)),
            "throughput_samples_per_sec": float(batch_size / (np.mean(latencies_us) / 1_000_000.0)),
        }
