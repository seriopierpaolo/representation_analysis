import os
import sys
import json
import time
import socket
import random
import platform
import datetime
import traceback
from typing import Dict, Any, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_manager.kitti_dataset_resources_analysis as kitti_dataset
from models.model_selector import model_selector
from utils.config_reader import get_config


# ============================================================
# Configuration
# ============================================================

REPRESENTATION = "cartesian_multi"
MODEL = "dino3"

KITTI_EVAL_SEQUENCES = ["02", "05", "06"]

CHECKPOINT_PATH = (
    "./representation_analysis/RESULTS/Trainings/"
    + REPRESENTATION
    + "/checkpoint/"
)

REPORT_DIR = "./representation_analysis/temp"
REPORT_PATH = os.path.join(REPORT_DIR, "kitti_inference_resource_analysis.json")

# Use "cuda:1" if available, otherwise fall back safely.
REQUESTED_DEVICE = "cuda:1"

# Warm-up batches are useful for stable GPU timing.
NUM_WARMUP_BATCHES = 3

# Store only aggregate statistics, not per-batch records.
SAVE_AGGREGATE_ONLY = True


# ============================================================
# Optional dependencies
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    PYNVML_AVAILABLE = False


# ============================================================
# Generic utility functions
# ============================================================

def bytes_to_mb(x):
    if x is None:
        return None
    return float(x) / (1024.0 ** 2)


def get_process_handle():
    if PSUTIL_AVAILABLE:
        return psutil.Process(os.getpid())
    return None


def get_rss_bytes(proc=None):
    if proc is not None:
        return int(proc.memory_info().rss)
    return None


def get_cpu_time_seconds(proc=None):
    if proc is not None:
        t = proc.cpu_times()
        return float(t.user + t.system)
    return float(time.process_time())


def summarize_values(values: List[float]) -> Dict[str, Any]:
    values = [float(v) for v in values if v is not None]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "sum": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


def summarize_gpu_vram_entries(entries: List[Any]):
    """
    Aggregates optional NVML GPU VRAM entries.

    Each entry is expected to be a list of dictionaries, one per GPU.
    """
    entries = [e for e in entries if e is not None]

    if len(entries) == 0:
        return None

    per_gpu = {}

    for entry in entries:
        for gpu in entry:
            idx = gpu["gpu_index"]

            if idx not in per_gpu:
                per_gpu[idx] = {
                    "gpu_index": idx,
                    "gpu_name": gpu["gpu_name"],
                    "vram_used_mb": [],
                    "vram_delta_mb": [],
                }

            if gpu.get("vram_used_mb") is not None:
                per_gpu[idx]["vram_used_mb"].append(gpu["vram_used_mb"])

            if gpu.get("vram_delta_mb") is not None:
                per_gpu[idx]["vram_delta_mb"].append(gpu["vram_delta_mb"])

    output = []

    for _, data in per_gpu.items():
        output.append({
            "gpu_index": data["gpu_index"],
            "gpu_name": data["gpu_name"],
            "vram_used_mb": summarize_values(data["vram_used_mb"]),
            "vram_delta_mb": summarize_values(data["vram_delta_mb"]),
        })

    return output


def get_nvml_gpu_snapshot():
    """
    Returns current NVIDIA VRAM usage using NVML.

    This is complementary to PyTorch CUDA memory statistics.
    PyTorch reports memory managed by PyTorch; NVML reports total process/system
    GPU memory usage visible to the driver.
    """
    if not PYNVML_AVAILABLE:
        return None

    try:
        output = []
        n_devices = pynvml.nvmlDeviceGetCount()

        for idx in range(n_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)

            if isinstance(name, bytes):
                name = name.decode("utf-8")

            output.append({
                "gpu_index": idx,
                "gpu_name": name,
                "vram_used_bytes": int(mem.used),
                "vram_used_mb": bytes_to_mb(mem.used),
                "vram_total_bytes": int(mem.total),
                "vram_total_mb": bytes_to_mb(mem.total),
                "vram_free_bytes": int(mem.free),
                "vram_free_mb": bytes_to_mb(mem.free),
            })

        return output

    except Exception:
        return None


def get_nvml_gpu_delta(before, after):
    if before is None or after is None:
        return None

    if len(before) != len(after):
        return None

    result = []

    for b, a in zip(before, after):
        delta = a["vram_used_bytes"] - b["vram_used_bytes"]

        result.append({
            "gpu_index": a["gpu_index"],
            "gpu_name": a["gpu_name"],
            "vram_used_before_mb": b["vram_used_mb"],
            "vram_used_after_mb": a["vram_used_mb"],
            "vram_used_mb": a["vram_used_mb"],
            "vram_delta_bytes": int(delta),
            "vram_delta_mb": bytes_to_mb(delta),
        })

    return result


def get_torch_cuda_memory(device):
    """
    Returns PyTorch CUDA memory stats for the selected device.

    allocated/reserved:
        Current memory tracked by PyTorch.

    max_allocated/max_reserved:
        Peak memory tracked by PyTorch since the last reset_peak_memory_stats().
    """
    if device.type != "cuda":
        return None

    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(device)),

        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "memory_reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(device)),

        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device)),

        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "max_memory_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device)),
    }


def resolve_device():
    if torch.cuda.is_available():
        requested = torch.device(REQUESTED_DEVICE)

        if requested.index is not None and requested.index < torch.cuda.device_count():
            return requested

        return torch.device("cuda:0")

    return torch.device("cpu")


# ============================================================
# Model loading
# ============================================================

def load_weights(path, model):
    ckpt_path = os.path.join(path, "model_best.pth.tar")

    if not os.path.isfile(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}")
        return model

    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    return model


# ============================================================
# Inference profiling
# ============================================================

def make_empty_inference_store():
    return {
        "num_batches": 0,
        "num_images": 0,

        "batch_size_values": [],

        "batch_wall_time_seconds": [],
        "batch_cpu_process_time_seconds": [],
        "batch_estimated_cpu_utilization_percent": [],

        "per_image_wall_time_seconds": [],
        "per_image_cpu_process_time_seconds": [],

        "ram_rss_before_mb": [],
        "ram_rss_after_mb": [],
        "ram_rss_delta_mb": [],

        "torch_cuda_memory_allocated_mb": [],
        "torch_cuda_memory_reserved_mb": [],
        "torch_cuda_max_memory_allocated_mb": [],
        "torch_cuda_max_memory_reserved_mb": [],

        "nvml_gpu_vram_entries": [],

        "input_shape_examples": [],
        "local_feature_shape_examples": [],
        "global_descriptor_shape_examples": [],
    }


def finalize_inference_summary(store):
    summary = {
        "num_batches": int(store["num_batches"]),
        "num_images": int(store["num_images"]),

        "batch_size": summarize_values(store["batch_size_values"]),

        "batch_wall_time_seconds": summarize_values(store["batch_wall_time_seconds"]),
        "batch_wall_time_ms": summarize_values(
            [v * 1000.0 for v in store["batch_wall_time_seconds"]]
        ),

        "batch_cpu_process_time_seconds": summarize_values(
            store["batch_cpu_process_time_seconds"]
        ),
        "batch_cpu_process_time_ms": summarize_values(
            [v * 1000.0 for v in store["batch_cpu_process_time_seconds"]]
        ),

        "batch_estimated_cpu_utilization_percent": summarize_values(
            store["batch_estimated_cpu_utilization_percent"]
        ),

        "per_image_wall_time_seconds": summarize_values(
            store["per_image_wall_time_seconds"]
        ),
        "per_image_wall_time_ms": summarize_values(
            [v * 1000.0 for v in store["per_image_wall_time_seconds"]]
        ),

        "per_image_cpu_process_time_seconds": summarize_values(
            store["per_image_cpu_process_time_seconds"]
        ),
        "per_image_cpu_process_time_ms": summarize_values(
            [v * 1000.0 for v in store["per_image_cpu_process_time_seconds"]]
        ),

        "ram_rss_before_mb": summarize_values(store["ram_rss_before_mb"]),
        "ram_rss_after_mb": summarize_values(store["ram_rss_after_mb"]),
        "ram_rss_delta_mb": summarize_values(store["ram_rss_delta_mb"]),

        "torch_cuda_memory_allocated_mb": summarize_values(
            store["torch_cuda_memory_allocated_mb"]
        ),
        "torch_cuda_memory_reserved_mb": summarize_values(
            store["torch_cuda_memory_reserved_mb"]
        ),
        "torch_cuda_max_memory_allocated_mb": summarize_values(
            store["torch_cuda_max_memory_allocated_mb"]
        ),
        "torch_cuda_max_memory_reserved_mb": summarize_values(
            store["torch_cuda_max_memory_reserved_mb"]
        ),

        "nvml_gpu_vram": summarize_gpu_vram_entries(
            store["nvml_gpu_vram_entries"]
        ),

        "input_shape_examples": store["input_shape_examples"],
        "local_feature_shape_examples": store["local_feature_shape_examples"],
        "global_descriptor_shape_examples": store["global_descriptor_shape_examples"],
    }

    mean_per_image = summary["per_image_wall_time_seconds"]["mean"]
    p95_per_image = summary["per_image_wall_time_seconds"]["p95"]
    max_per_image = summary["per_image_wall_time_seconds"]["max"]

    summary["mean_fps_single_stream_equivalent"] = (
        float(1.0 / mean_per_image)
        if mean_per_image is not None and mean_per_image > 0
        else None
    )

    summary["p95_fps_single_stream_equivalent"] = (
        float(1.0 / p95_per_image)
        if p95_per_image is not None and p95_per_image > 0
        else None
    )

    summary["worst_case_fps_single_stream_equivalent"] = (
        float(1.0 / max_per_image)
        if max_per_image is not None and max_per_image > 0
        else None
    )

    return summary


def warmup_model(model, data_loader, device, num_warmup_batches):
    """
    Runs a few forward passes before profiling.

    This helps reduce one-time CUDA/kernel initialization effects.
    Warm-up is not included in the reported inference timings.
    """
    if num_warmup_batches <= 0:
        return

    model.eval()

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(data_loader):
            if batch_idx >= num_warmup_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)


def infer_with_profiling(eval_set, seq, device, opt, model, return_local_feats=False):
    """
    Runs inference and returns:
    - global descriptors
    - optional local features
    - aggregate-only inference resource summary
    """
    test_data_loader = DataLoader(
        dataset=eval_set,
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    model.to(device)

    warmup_model(
        model=model,
        data_loader=test_data_loader,
        device=device,
        num_warmup_batches=NUM_WARMUP_BATCHES,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    proc = get_process_handle()
    store = make_empty_inference_store()

    all_global_descs = []
    all_local_feats = []

    sequence_wall_start = time.perf_counter()
    sequence_cpu_start = get_cpu_time_seconds(proc)
    sequence_rss_start = get_rss_bytes(proc)
    sequence_nvml_start = get_nvml_gpu_snapshot()

    with torch.no_grad():
        for _, (imgs, _) in enumerate(tqdm(test_data_loader, disable=False)):
            batch_size = int(imgs.shape[0])

            if len(store["input_shape_examples"]) == 0:
                store["input_shape_examples"].append({
                    "shape": list(imgs.shape),
                    "dtype": str(imgs.dtype),
                })

            imgs = imgs.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            rss_before = get_rss_bytes(proc)
            cpu_before = get_cpu_time_seconds(proc)
            nvml_before = get_nvml_gpu_snapshot()

            wall_start = time.perf_counter()

            local_feat, global_desc = model(imgs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            wall_time = time.perf_counter() - wall_start
            cpu_time = get_cpu_time_seconds(proc) - cpu_before

            rss_after = get_rss_bytes(proc)
            nvml_after = get_nvml_gpu_snapshot()
            torch_cuda_mem = get_torch_cuda_memory(device)

            cpu_util = (
                100.0 * cpu_time / wall_time
                if wall_time > 0
                else None
            )

            store["num_batches"] += 1
            store["num_images"] += batch_size

            store["batch_size_values"].append(batch_size)

            store["batch_wall_time_seconds"].append(float(wall_time))
            store["batch_cpu_process_time_seconds"].append(float(cpu_time))

            if cpu_util is not None:
                store["batch_estimated_cpu_utilization_percent"].append(float(cpu_util))

            store["per_image_wall_time_seconds"].append(float(wall_time / batch_size))
            store["per_image_cpu_process_time_seconds"].append(float(cpu_time / batch_size))

            if rss_before is not None:
                store["ram_rss_before_mb"].append(bytes_to_mb(rss_before))
            if rss_after is not None:
                store["ram_rss_after_mb"].append(bytes_to_mb(rss_after))
            if rss_before is not None and rss_after is not None:
                store["ram_rss_delta_mb"].append(bytes_to_mb(rss_after - rss_before))

            if torch_cuda_mem is not None:
                store["torch_cuda_memory_allocated_mb"].append(
                    torch_cuda_mem["memory_allocated_mb"]
                )
                store["torch_cuda_memory_reserved_mb"].append(
                    torch_cuda_mem["memory_reserved_mb"]
                )
                store["torch_cuda_max_memory_allocated_mb"].append(
                    torch_cuda_mem["max_memory_allocated_mb"]
                )
                store["torch_cuda_max_memory_reserved_mb"].append(
                    torch_cuda_mem["max_memory_reserved_mb"]
                )

            nvml_delta = get_nvml_gpu_delta(nvml_before, nvml_after)
            if nvml_delta is not None:
                store["nvml_gpu_vram_entries"].append(nvml_delta)

            if len(store["local_feature_shape_examples"]) == 0:
                if isinstance(local_feat, torch.Tensor):
                    store["local_feature_shape_examples"].append({
                        "shape": list(local_feat.shape),
                        "dtype": str(local_feat.dtype),
                    })

            if len(store["global_descriptor_shape_examples"]) == 0:
                if isinstance(global_desc, torch.Tensor):
                    store["global_descriptor_shape_examples"].append({
                        "shape": list(global_desc.shape),
                        "dtype": str(global_desc.dtype),
                    })

            all_global_descs.append(global_desc.detach().cpu().numpy())

            if return_local_feats:
                all_local_feats.append(local_feat.detach().cpu().numpy())

    sequence_wall_time = time.perf_counter() - sequence_wall_start
    sequence_cpu_time = get_cpu_time_seconds(proc) - sequence_cpu_start
    sequence_rss_end = get_rss_bytes(proc)
    sequence_nvml_end = get_nvml_gpu_snapshot()

    global_descs = np.concatenate(all_global_descs, axis=0)

    inference_summary = finalize_inference_summary(store)

    inference_summary["sequence_level"] = {
        "sequence": seq,
        "total_inference_wall_time_seconds": float(sequence_wall_time),
        "total_inference_cpu_process_time_seconds": float(sequence_cpu_time),
        "estimated_cpu_utilization_percent": (
            float(100.0 * sequence_cpu_time / sequence_wall_time)
            if sequence_wall_time > 0
            else None
        ),
        "ram_rss_start_mb": bytes_to_mb(sequence_rss_start),
        "ram_rss_end_mb": bytes_to_mb(sequence_rss_end),
        "ram_rss_delta_mb": (
            bytes_to_mb(sequence_rss_end - sequence_rss_start)
            if sequence_rss_start is not None and sequence_rss_end is not None
            else None
        ),
        "nvml_gpu_vram_sequence_delta": get_nvml_gpu_delta(
            sequence_nvml_start,
            sequence_nvml_end,
        ),
        "descriptor_array": {
            "shape": list(global_descs.shape),
            "dtype": str(global_descs.dtype),
            "nbytes": int(global_descs.nbytes),
            "nbytes_mb": bytes_to_mb(global_descs.nbytes),
        },
    }

    if return_local_feats:
        local_feats = np.concatenate(all_local_feats, axis=0)

        inference_summary["sequence_level"]["local_feature_array"] = {
            "shape": list(local_feats.shape),
            "dtype": str(local_feats.dtype),
            "nbytes": int(local_feats.nbytes),
            "nbytes_mb": bytes_to_mb(local_feats.nbytes),
        }

        return local_feats, global_descs, inference_summary

    return global_descs, inference_summary


# ============================================================
# Report helpers
# ============================================================

def get_environment_metadata(opt, device, model):
    metadata = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),

        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "selected_device": str(device),

        "psutil_available": PSUTIL_AVAILABLE,
        "pynvml_available": PYNVML_AVAILABLE,
        "nvml_gpu_snapshot_at_start": get_nvml_gpu_snapshot(),

        "configuration": {
            "dataset": "KITTI",
            "eval_sequences": KITTI_EVAL_SEQUENCES,
            "representation": REPRESENTATION,
            "model": MODEL,
            "checkpoint_path": CHECKPOINT_PATH,
            "batch_size": opt.cacheBatchSize,
            "num_dataloader_workers": opt.threads,
            "num_warmup_batches": NUM_WARMUP_BATCHES,
            "report_path": REPORT_PATH,
        },

        "metric_definitions": {
            "batch_wall_time_ms": (
                "Forward-pass elapsed time per batch, measured with time.perf_counter(). "
                "CUDA synchronization is applied before stopping the timer."
            ),
            "per_image_wall_time_ms": (
                "batch_wall_time_ms divided by batch size."
            ),
            "batch_cpu_process_time_ms": (
                "User + system CPU time consumed by the Python process during each model forward pass."
            ),
            "ram_rss_delta_mb": (
                "Change in resident set size before and after each forward pass."
            ),
            "torch_cuda_max_memory_allocated_mb": (
                "Peak CUDA memory allocated by PyTorch during the measured forward pass."
            ),
            "torch_cuda_max_memory_reserved_mb": (
                "Peak CUDA memory reserved by PyTorch during the measured forward pass."
            ),
            "nvml_gpu_vram": (
                "Optional NVIDIA VRAM usage from NVML. This may include memory outside PyTorch."
            ),
            "faiss_search_time_ms_per_query": (
                "Time needed by FAISS to retrieve the nearest database descriptor per query."
            ),
        },

        "reviewer_facing_notes": [
            "This report is restricted to KITTI only.",
            "Inference latency excludes dataloader time and descriptor evaluation time; it measures the neural-network forward pass.",
            "FAISS retrieval time is reported separately from model inference.",
            "Warm-up batches are excluded from timing.",
            "GPU memory is reported using PyTorch CUDA statistics when CUDA is available. NVML VRAM is reported only if pynvml is available.",
        ],
    }

    if torch.cuda.is_available():
        cuda_devices = []

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append({
                "index": idx,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "total_memory_mb": bytes_to_mb(props.total_memory),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })

        metadata["cuda_devices"] = cuda_devices

    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        metadata["system_memory"] = {
            "total_bytes": int(vm.total),
            "total_mb": bytes_to_mb(vm.total),
            "available_bytes": int(vm.available),
            "available_mb": bytes_to_mb(vm.available),
        }

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    metadata["model_parameters"] = {
        "num_trainable": int(num_trainable),
        "num_total": int(num_total),
        "num_trainable_million": float(num_trainable / 1e6),
        "num_total_million": float(num_total / 1e6),
    }

    return metadata


def finalize_overall_kitti_summary(sequence_reports):
    """
    Builds an overall KITTI summary from per-sequence aggregate reports.
    """
    inference_per_image_mean_ms = []
    inference_per_image_p95_ms = []
    inference_per_image_p99_ms = []
    inference_per_image_max_ms = []

    inference_batch_mean_ms = []
    torch_peak_allocated_mb = []
    torch_peak_reserved_mb = []

    faiss_search_per_query_ms = []
    faiss_total_search_ms = []
    recall_at1 = []
    f1 = []
    auc = []

    num_images = 0

    for seq_report in sequence_reports:
        inf = seq_report["inference"]
        ev = seq_report["evaluation"]

        num_images += inf["num_images"]

        inference_per_image_mean_ms.append(inf["per_image_wall_time_ms"]["mean"])
        inference_per_image_p95_ms.append(inf["per_image_wall_time_ms"]["p95"])
        inference_per_image_p99_ms.append(inf["per_image_wall_time_ms"]["p99"])
        inference_per_image_max_ms.append(inf["per_image_wall_time_ms"]["max"])

        inference_batch_mean_ms.append(inf["batch_wall_time_ms"]["mean"])

        torch_peak_allocated_mb.append(
            inf["torch_cuda_max_memory_allocated_mb"]["max"]
        )
        torch_peak_reserved_mb.append(
            inf["torch_cuda_max_memory_reserved_mb"]["max"]
        )

        faiss_search_per_query_ms.append(
            ev["faiss"]["search_time_per_query_ms"]
        )
        faiss_total_search_ms.append(
            ev["faiss"]["total_search_time_ms"]
        )

        recall_at1.append(ev["metrics"]["recall_at1"])
        f1.append(ev["metrics"]["max_f1"])
        auc.append(ev["metrics"]["auc"])

    overall = {
        "num_sequences": int(len(sequence_reports)),
        "num_images": int(num_images),

        "inference_per_image_wall_time_ms": {
            "mean_across_sequences": summarize_values(inference_per_image_mean_ms),
            "p95_across_sequences": summarize_values(inference_per_image_p95_ms),
            "p99_across_sequences": summarize_values(inference_per_image_p99_ms),
            "max_across_sequences": summarize_values(inference_per_image_max_ms),
        },

        "inference_batch_wall_time_ms": {
            "mean_across_sequences": summarize_values(inference_batch_mean_ms),
        },

        "torch_cuda_peak_memory_mb": {
            "max_allocated_across_sequences": summarize_values(torch_peak_allocated_mb),
            "max_reserved_across_sequences": summarize_values(torch_peak_reserved_mb),
        },

        "faiss_search_time_per_query_ms": summarize_values(faiss_search_per_query_ms),
        "faiss_total_search_time_ms": summarize_values(faiss_total_search_ms),

        "retrieval_metrics": {
            "recall_at1": summarize_values(recall_at1),
            "max_f1": summarize_values(f1),
            "auc": summarize_values(auc),
        },
    }

    mean_per_image_ms = overall["inference_per_image_wall_time_ms"]["mean_across_sequences"]["mean"]

    if mean_per_image_ms is not None and mean_per_image_ms > 0:
        overall["mean_fps_single_stream_equivalent"] = float(1000.0 / mean_per_image_ms)
    else:
        overall["mean_fps_single_stream_equivalent"] = None

    return overall


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)

    opt = get_config()
    device = resolve_device()

    print("=" * 80)
    print("KITTI-only inference resource analysis")
    print(f"Model: {MODEL}")
    print(f"Representation: {REPRESENTATION}")
    print(f"Device: {device}")
    print(f"Batch size: {opt.cacheBatchSize}")
    print(f"Dataloader workers: {opt.threads}")
    print(f"Report path: {REPORT_PATH}")
    print("=" * 80)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True

    print("===> Building model")
    model, projection_dimension = model_selector(MODEL, device)
    model = model.to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {num_trainable / 1e6:.2f}M / {num_total / 1e6:.2f}M")

    model = load_weights(CHECKPOINT_PATH, model)
    model.eval()

    sequence_reports = []

    experiment_wall_start = time.perf_counter()

    for seq in KITTI_EVAL_SEQUENCES:
        try:
            print("=" * 80)
            print(f"Processing KITTI sequence {seq}")
            print("=" * 80)

            test_set = kitti_dataset.InferDataset(
                seq=seq,
                representation=REPRESENTATION,
            )

            global_descs, inference_summary = infer_with_profiling(
                eval_set=test_set,
                seq=seq,
                device=device,
                opt=opt,
                model=model,
                return_local_feats=False,
            )

            recall_at1, max_f1, auc, evaluation_summary = kitti_dataset.evaluateResults(
                seq=seq,
                global_descs=global_descs,
                dataset=test_set,
                return_metrics=True,
            )

            print(f"{seq} - Recall@1: {recall_at1 * 100:.2f}")
            print(f"{seq} - F1: {max_f1 * 100:.2f}")
            print(f"{seq} - AUC: {auc * 100:.2f}")

            sequence_reports.append({
                "sequence": seq,
                "num_images": int(len(test_set)),
                "db_split_index": int(test_set.db_split_index),
                "query_count": int(len(test_set) - test_set.db_split_index),
                "inference": inference_summary,
                "evaluation": evaluation_summary,
            })

        except Exception as exc:
            sequence_reports.append({
                "sequence": seq,
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })

    experiment_wall_time = time.perf_counter() - experiment_wall_start

    successful_sequence_reports = [
        r for r in sequence_reports
        if r.get("status") != "failed"
    ]

    report = {
        "report_type": "kitti_model_inference_resource_analysis",
        "environment": get_environment_metadata(opt, device, model),
        "summary": {
            "total_experiment_wall_time_seconds": float(experiment_wall_time),
            "num_sequences_requested": int(len(KITTI_EVAL_SEQUENCES)),
            "num_sequences_successful": int(len(successful_sequence_reports)),
            "num_sequences_failed": int(len(sequence_reports) - len(successful_sequence_reports)),
            "overall_kitti": finalize_overall_kitti_summary(successful_sequence_reports)
            if len(successful_sequence_reports) > 0
            else None,
        },
        "sequences": sequence_reports,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("KITTI inference resource analysis complete.")
    print(f"JSON report written to: {REPORT_PATH}")
    print("=" * 80)