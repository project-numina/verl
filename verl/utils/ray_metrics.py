import os

# Set WANDB_MODE before importing wandb
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

import ray
import wandb
import psutil
import torch
from typing import Dict, Any
import time
import socket


class RayMetricsReporter:
    def __init__(self, wandb_project: str, wandb_run_name: str = None):
        """Initialize the Ray metrics reporter with wandb configuration."""
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.start_time = time.time()

        if not wandb.run:
            node_name = socket.gethostname()
            run_name = f"{wandb_run_name}_{node_name}" if wandb_run_name else f"ray_metrics_{node_name}"
            mode = os.environ.get("WANDB_MODE", "offline")

            # Initialize wandb
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                mode=mode,
                config={
                    "node_name": node_name,
                    "ray_node_id": ray.get_runtime_context().node_id.hex(),
                    "ray_job_id": ray.get_runtime_context().job_id.hex(),
                }
            )

    def collect_node_metrics(self) -> Dict[str, Any]:
        """Collect metrics for the current node."""
        metrics = {}

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()

        metrics.update({
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
        })

        # GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics.update({
                    f"gpu_{i}_memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    f"gpu_{i}_memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    # You may need to install `nvidia-ml-py3` to use `torch.cuda.utilization`
                    # f"gpu_{i}_utilization": torch.cuda.utilization(i),
                })

        return metrics

    def collect_ray_metrics(self) -> Dict[str, Any]:
        """Collect Ray-specific metrics."""
        metrics = {}
        resources = ray.available_resources()

        metrics.update({
            "ray_cpu_available": resources.get("CPU", 0),
            "ray_gpu_available": resources.get("GPU", 0),
            "ray_memory_available_gb": resources.get("memory", 0) / (1024**3),
        })

        nodes = ray.nodes()
        metrics["ray_nodes_active"] = len([n for n in nodes if n["Alive"]])

        return metrics

    def log_metrics(self):
        """Collect and log all metrics to wandb."""
        metrics = {}

        # Collect metrics
        node_metrics = self.collect_node_metrics()
        ray_metrics = self.collect_ray_metrics()
        metrics.update({f"node/{k}": v for k, v in node_metrics.items()})
        metrics.update({f"ray/{k}": v for k, v in ray_metrics.items()})
        metrics["timestamp"] = time.time() - self.start_time

        # Log to wandb
        wandb.log(metrics)

    def finish(self):
        """Close the W&B run cleanly."""
        if wandb.run:
            wandb.finish()
