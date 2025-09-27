#   Copyright 2025 DataXight, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


"""
Monkey patch for daft.runners.flotilla to create one worker per CPU core.

This module provides a patched version of start_ray_workers that creates
one worker per CPU core instead of one worker per node for better parallelization.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daft.daft import RaySwordfishWorker

try:
    import ray
except ImportError:
    raise ImportError("Ray is required for flotilla patches")  # noqa: B904


def start_ray_workers_per_cpu(existing_worker_ids: list[str]) -> list["RaySwordfishWorker"]:
    """
    Create one RaySwordfishWorker per CPU core instead of one per node.

    This is a patched version of daft.runners.flotilla.start_ray_workers that
    provides better CPU utilization by creating individual workers for each CPU core.

    Args:
        existing_worker_ids: List of existing worker IDs to avoid duplicates

    Returns:
        List of RaySwordfishWorker instances, one per CPU core
    """
    # Import these here to avoid circular imports
    from daft.daft import RaySwordfishWorker
    from daft.runners.flotilla import RaySwordfishActor, RaySwordfishActorHandle

    max_workers = int(os.environ.get("MAX_WORKERS", 0))
    if max_workers == 0:
        max_workers = float("inf")

    handles = []
    for node in ray.nodes():
        if (
            "Resources" in node
            and "CPU" in node["Resources"]
            and "memory" in node["Resources"]
            and node["Resources"]["CPU"] > 0
            and node["Resources"]["memory"] > 0
            and node["NodeID"] not in existing_worker_ids
        ):
            num_cpus = int(node["Resources"]["CPU"])
            num_gpus = int(node["Resources"].get("GPU", 0))
            total_memory = int(node["Resources"]["memory"])

            # Create one worker per CPU core
            for cpu_idx in range(min(num_cpus, max_workers)):
                worker_id = f"{node['NodeID']}_cpu_{cpu_idx}"

                # Skip if this specific worker already exists
                if worker_id in existing_worker_ids:
                    continue

                actor = RaySwordfishActor.options(  # type: ignore
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    ),
                    num_cpus=1,  # Allocate 1 CPU per worker
                ).remote(
                    num_cpus=1,  # Each worker uses 1 CPU
                    num_gpus=num_gpus if cpu_idx == 0 else 0,  # Only first worker gets GPUs
                )
                actor_handle = RaySwordfishActorHandle(actor)
                handles.append(
                    RaySwordfishWorker(
                        worker_id,
                        actor_handle,
                        1,  # 1 CPU per worker
                        num_gpus if cpu_idx == 0 else 0,  # Only first worker gets GPUs
                        total_memory // num_cpus,  # Distribute memory evenly across workers
                    )
                )

    return handles


def apply_flotilla_patches():
    """
    Apply monkey patches to daft.runners.flotilla module.

    This function should be called before importing daft to ensure
    the patches are applied correctly.
    """
    try:
        import daft.runners.flotilla as flotilla_module

        # Store original function for potential rollback
        if not hasattr(flotilla_module, "_original_start_ray_workers"):
            flotilla_module._original_start_ray_workers = flotilla_module.start_ray_workers

        # Apply the patch
        flotilla_module.start_ray_workers = start_ray_workers_per_cpu

        print("✓ Applied daft flotilla patch: start_ray_workers now creates one worker per CPU")

    except ImportError:
        print("⚠ daft.runners.flotilla not available for patching")


def rollback_flotilla_patches():
    """
    Rollback flotilla patches to original daft implementation.
    """
    try:
        import daft.runners.flotilla as flotilla_module

        if hasattr(flotilla_module, "_original_start_ray_workers"):
            flotilla_module.start_ray_workers = flotilla_module._original_start_ray_workers
            delattr(flotilla_module, "_original_start_ray_workers")
            print("✓ Rolled back daft flotilla patches")
        else:
            print("⚠ No patches to rollback")

    except ImportError:
        print("⚠ daft.runners.flotilla not available for rollback")
