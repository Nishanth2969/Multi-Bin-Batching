"""
Run manifest schema for reproducible MBB experiments.

Defines configuration for benchmark runs including workload, policy,
and environment settings.
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class DecodeConfig:
    """Decode/generation control parameters."""
    
    temperature: float = 0.0
    max_new_tokens: int = 512
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42
    stop_tokens: List[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    
    policy: str = "continuous"  # 'continuous' or 'multi_bin'
    num_bins: int = 3
    bin_edges: Optional[List[int]] = None
    use_equal_mass_bins: bool = False
    warmup_samples: int = 100
    predictor_type: str = "heuristic"  # 'heuristic', 'oracle', 'model'
    selection_policy: str = "round_robin"  # 'round_robin', 'weighted'
    starvation_ms: int = 500
    kv_budget_tokens: int = 8192
    max_batched_tokens: int = 2048


@dataclass
class WorkloadConfig:
    """Workload definition."""
    
    name: str
    dataset_path: str
    arrival_rate: float  # requests per second
    duration_s: int = 300  # 5 minutes default
    replay_trace: Optional[str] = None  # Path to arrival trace file
    num_requests: Optional[int] = None  # If set, overrides duration


@dataclass
class SystemConfig:
    """System/hardware configuration."""
    
    gpu_device_id: int = 0
    model_name: str = "meta-llama/Llama-2-7b-hf"
    model_path: Optional[str] = None
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192


@dataclass
class RunManifest:
    """Complete run configuration manifest."""
    
    run_id: str
    description: str
    workload: WorkloadConfig
    scheduler: SchedulerConfig
    decode: DecodeConfig
    system: SystemConfig
    output_dir: str = "results"
    replicate_id: int = 0
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """Save to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'RunManifest':
        """Load from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'RunManifest':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunManifest':
        """Construct from dictionary."""
        return cls(
            run_id=data['run_id'],
            description=data['description'],
            workload=WorkloadConfig(**data['workload']),
            scheduler=SchedulerConfig(**data['scheduler']),
            decode=DecodeConfig(**data['decode']),
            system=SystemConfig(**data['system']),
            output_dir=data.get('output_dir', 'results'),
            replicate_id=data.get('replicate_id', 0),
            git_commit=data.get('git_commit'),
            timestamp=data.get('timestamp')
        )


def generate_baseline_manifest(workload_name: str = "mixed", 
                               arrival_rate: float = 100.0) -> RunManifest:
    """Generate baseline continuous batching manifest.
    
    Args:
        workload_name: Workload mix name
        arrival_rate: Requests per second
        
    Returns:
        RunManifest for baseline run
    """
    return RunManifest(
        run_id=f"baseline_{workload_name}_r{int(arrival_rate)}",
        description=f"Baseline continuous batching on {workload_name} workload",
        workload=WorkloadConfig(
            name=workload_name,
            dataset_path=f"data/{workload_name}.txt",
            arrival_rate=arrival_rate,
            duration_s=300
        ),
        scheduler=SchedulerConfig(
            policy="continuous",
            num_bins=1  # k=1 is equivalent to baseline
        ),
        decode=DecodeConfig(
            temperature=0.0,
            max_new_tokens=512,
            seed=42
        ),
        system=SystemConfig(
            model_name="microsoft/Phi-3.5-mini-instruct",
            dtype="float16"
        )
    )


def generate_mbb_manifest(workload_name: str = "mixed", 
                         arrival_rate: float = 100.0,
                         num_bins: int = 3,
                         use_equal_mass: bool = True) -> RunManifest:
    """Generate multi-bin batching manifest.
    
    Args:
        workload_name: Workload mix name
        arrival_rate: Requests per second
        num_bins: Number of bins
        use_equal_mass: Use equal-mass quantile bins
        
    Returns:
        RunManifest for MBB run
    """
    return RunManifest(
        run_id=f"mbb_k{num_bins}_{workload_name}_r{int(arrival_rate)}",
        description=f"Multi-bin batching (k={num_bins}) on {workload_name} workload",
        workload=WorkloadConfig(
            name=workload_name,
            dataset_path=f"data/{workload_name}.txt",
            arrival_rate=arrival_rate,
            duration_s=300
        ),
        scheduler=SchedulerConfig(
            policy="multi_bin",
            num_bins=num_bins,
            bin_edges=[128, 512] if num_bins == 3 else None,
            use_equal_mass_bins=use_equal_mass,
            warmup_samples=100 if use_equal_mass else 0,
            predictor_type="heuristic"
        ),
        decode=DecodeConfig(
            temperature=0.0,
            max_new_tokens=512,
            seed=42  # Same seed as baseline for parity
        ),
        system=SystemConfig(
            model_name="microsoft/Phi-3.5-mini-instruct",
            dtype="float16"
        )
    )


def generate_ablation_manifests(workload_name: str = "mixed") -> List[RunManifest]:
    """Generate manifests for ablation study.
    
    Varies: k ∈ {1,2,3,4,8}, bin edges (static vs equal-mass), predictor
    
    Args:
        workload_name: Workload mix name
        
    Returns:
        List of RunManifest objects
    """
    manifests = []
    arrival_rate = 100.0
    
    # Vary number of bins
    for k in [1, 2, 3, 4, 8]:
        # Static edges
        manifest = generate_mbb_manifest(
            workload_name=workload_name,
            arrival_rate=arrival_rate,
            num_bins=k,
            use_equal_mass=False
        )
        manifest.run_id = f"ablation_k{k}_static_{workload_name}"
        manifests.append(manifest)
        
        # Equal-mass edges
        manifest = generate_mbb_manifest(
            workload_name=workload_name,
            arrival_rate=arrival_rate,
            num_bins=k,
            use_equal_mass=True
        )
        manifest.run_id = f"ablation_k{k}_equalmass_{workload_name}"
        manifests.append(manifest)
    
    # Oracle predictor with k=3
    manifest = generate_mbb_manifest(
        workload_name=workload_name,
        arrival_rate=arrival_rate,
        num_bins=3,
        use_equal_mass=True
    )
    manifest.scheduler.predictor_type = "oracle"
    manifest.run_id = f"ablation_oracle_{workload_name}"
    manifests.append(manifest)
    
    return manifests


if __name__ == "__main__":
    # Example: Generate and save manifests
    import os
    
    output_dir = "bench/manifests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline
    baseline = generate_baseline_manifest("mixed", 100.0)
    baseline.to_yaml(f"{output_dir}/baseline_mixed.yaml")
    print(f"Generated: {output_dir}/baseline_mixed.yaml")
    
    # MBB k=3
    mbb = generate_mbb_manifest("mixed", 100.0, num_bins=3)
    mbb.to_yaml(f"{output_dir}/mbb_k3_mixed.yaml")
    print(f"Generated: {output_dir}/mbb_k3_mixed.yaml")
    
    # Ablation study
    ablations = generate_ablation_manifests("mixed")
    for manifest in ablations:
        filepath = f"{output_dir}/{manifest.run_id}.yaml"
        manifest.to_yaml(filepath)
        print(f"Generated: {filepath}")
    
    print(f"\nGenerated {1 + 1 + len(ablations)} manifest files")

