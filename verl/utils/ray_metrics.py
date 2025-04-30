import ray
import wandb
import psutil
import torch
from typing import Dict, Any
import time
import os
import socket

class RayMetricsReporter:
    def __init__(self, wandb_project: str, wandb_run_name: str = None):
        """Initialize the Ray metrics reporter with wandb configuration."""
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.start_time = time.time()
        
        # Initialize wandb if not already initialized
        if not wandb.run:
            # Set wandb to offline mode if WANDB_MODE is not set
            if "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "offline"
            
            # Get node name for unique run identification
            node_name = socket.gethostname()
            
            # Initialize wandb with offline mode and unique run name per node
            wandb.init(
                project=wandb_project,
                name=f"{wandb_run_name}_{node_name}" if wandb_run_name else f"ray_metrics_{node_name}",
                mode="offline" if os.environ.get("WANDB_MODE") == "offline" else "online",
                config={
                    "node_name": node_name,
                    "ray_node_id": ray.get_runtime_context().node_id.hex(),
                    "ray_job_id": ray.get_runtime_context().job_id.hex(),
                }
            )
            
            # Ensure .wandb file is created
            if os.environ.get("WANDB_MODE") == "offline":
                wandb.run.temp.dir.flush()
                wandb.run.temp.dir.close()
                # Force creation of .wandb file
                wandb.run._init_run()
    
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
                    f"gpu_{i}_utilization": torch.cuda.utilization(i),
                })
        
        return metrics
    
    def collect_ray_metrics(self) -> Dict[str, Any]:
        """Collect Ray-specific metrics."""
        metrics = {}
        
        # Get Ray cluster resources
        resources = ray.available_resources()
        metrics.update({
            "ray_cpu_available": resources.get("CPU", 0),
            "ray_gpu_available": resources.get("GPU", 0),
            "ray_memory_available_gb": resources.get("memory", 0) / (1024**3),
        })
        
        # Get Ray worker status
        nodes = ray.nodes()
        metrics["ray_nodes_active"] = len([n for n in nodes if n["Alive"]])
        
        return metrics
    
    def log_metrics(self):
        """Collect and log all metrics to wandb."""
        metrics = {}
        
        # Collect node metrics
        node_metrics = self.collect_node_metrics()
        metrics.update({f"node/{k}": v for k, v in node_metrics.items()})
        
        # Collect Ray metrics
        ray_metrics = self.collect_ray_metrics()
        metrics.update({f"ray/{k}": v for k, v in ray_metrics.items()})
        
        # Add timestamp
        metrics["timestamp"] = time.time() - self.start_time
        
        # Log to wandb
        wandb.log(metrics)
        
        # Force sync if in offline mode
        if os.environ.get("WANDB_MODE") == "offline":
            wandb.run.temp.dir.flush()
            wandb.run.temp.dir.close()
            # Ensure .wandb file is updated
            wandb.run._init_run() 