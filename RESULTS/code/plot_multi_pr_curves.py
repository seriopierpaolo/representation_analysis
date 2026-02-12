# Optimized and cleaned version of your evaluation script.
# Imported functions are assumed correct and untouched.

import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import isfile, join

import dataset_manager.kitti_dataset as kitti_dataset
import dataset_manager.nclt_dataset as nclt_dataset
import dataset_manager.helilpr_dataset as helilpr_dataset
import dataset_manager.toyota_dataset as toyota_dataset
from models.model_selector import model_selector

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend


def infer(model, eval_set, device, batch_size, num_workers, return_local_feats=False):
    loader = DataLoader(eval_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()
    model.to(device)

    total_time, batches = 0.0, 0
    all_global, all_local = [], []

    with torch.no_grad():
        for imgs, _ in tqdm(loader, disable=True):
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            local_feat, global_desc = model(imgs)
            total_time += time.perf_counter() - t0
            batches += 1

            all_global.append(global_desc.cpu().numpy())
            if return_local_feats:
                all_local.append(local_feat.cpu().numpy())

    print(f"Avg inference time per batch: {total_time / max(batches,1):.4f}s")

    global_concat = np.concatenate(all_global, axis=0)
    if return_local_feats:
        return np.concatenate(all_local, axis=0), global_concat
    return global_concat


def load_weights(path, model):
    ckpt_path = os.path.join(path, 'model_best.pth.tar')
    if not os.path.isfile(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}")
        return model
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model


def plot_pr_curve(precisions, recalls, title, save_path=None):
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved PR curve: {save_path}")
    plt.close()


# ======================================================================
# =========================== MAIN SCRIPT ===============================
# ======================================================================

if __name__ == "__main__":

    device = torch.device("cuda:1")

    REPRESENTATIONS = ['cartesian_multi','polar_multi', "range_multi", "fv_multi"]

    DATASETS = {
        "KITTI": {
            "use": False,
            "seqs": ['02', '05', '06'],
            "loader": lambda seq, rep: kitti_dataset.InferDataset(seq=seq, representation=rep),
            "eval": kitti_dataset.evaluateResults_pr_only
        },
        "NCLT": {
            "use": False,
            "seqs": ['2012-01-15','2012-02-04','2012-06-15','2013-02-23'],
            "loader": lambda seq, rep: nclt_dataset.InferDataset(seq=seq, representation=rep),
            "eval": nclt_dataset.evaluateResults_pr_only
        },
        "HELILPR": {
            "use": True,
            "seqs": ['rb02','rb03'],
            "loader": lambda seq, rep: helilpr_dataset.InferDataset(seq=seq, representation=rep),
            "eval": helilpr_dataset.evaluateResults_pr_only
        },
        "TOYOTA": {
            "use": True,
            "seqs": ['00'],
            "loader": lambda seq, rep: toyota_dataset.InferDataset(seq=seq, representation=rep),
            "eval": toyota_dataset.evaluateResults_pr_only
        }
    }

    # --------------------------------------------------
    # Storage for multi-representation PR curves
    # PR_DATA[dataset][seq][rep] = (P, R, Recall@1)
    # --------------------------------------------------

    PR_DATA = {
        ds: {seq: {} for seq in DATASETS[ds]["seqs"]}
        for ds in DATASETS
    }

    # Reproducibility
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed(seed)

    MODEL = "dino3vlad"
    model, proj_dim = model_selector(MODEL, device)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ======================================================
    # ============ ORIGINAL LOOP STRUCTURE KEPT ============
    # ======================================================

    for rep in REPRESENTATIONS:
        print(f"\n========== Representation: {rep} ==========")
        model = load_weights(f"./representation_analysis/RESULTS/Trainings/{rep}/checkpoint/", model)

        for dataset_name, cfg in DATASETS.items():
            if not cfg["use"]:
                continue

            print(f"\n--- Dataset: {dataset_name} ---")

            # For NCLT special DB handling
            nclt_db_descs = None
            nclt_db_set = None

            for seq in cfg["seqs"]:
                print(f"Processing seq {seq}")

                test_set = cfg["loader"](seq, rep)
                global_descs = infer(model, test_set, device, batch_size=32, num_workers=4)

                # ---------- NCLT SPECIAL DB/QUERY LOGIC ----------
                if dataset_name == "NCLT":
                    # DB sequence — never produce PR
                    if seq == cfg["seqs"][0]:
                        nclt_db_descs = global_descs
                        nclt_db_set = test_set
                        print(f"Using {seq} as NCLT database")
                        continue

                    descriptors = [nclt_db_descs, global_descs]
                    datasets = [nclt_db_set, test_set]

                    results = cfg["eval"](seq, descriptors, datasets)

                else:
                    results = cfg["eval"](seq, global_descs, test_set)

                # Parse results
                recall_at1 = results[0]
                if len(results) > 2:
                    P = results[1]
                    R = results[2]
                else:
                    P, R = None, None

                print(f"Seq {seq} Recall@1 = {recall_at1:.2f}")

                # Store PR data for multi-representation plotting
                # Skip NCLT DB sequence
                if dataset_name == "NCLT" and seq == cfg["seqs"][0]:
                    continue

                if P is not None and R is not None:
                    PR_DATA[dataset_name][seq][rep] = (P, R, recall_at1)

    # ======================================================
    # === PLOT MULTI-REPRESENTATION PR CURVES PER SEQUENCE ==
    # ======================================================

    print("\n=========== Generating multi-representation PR plots ===========")

    # --- START OF MODIFICATION ---
    # 1. Define the mapping for the legend labels
    LEGEND_MAPPING = {
        "cartesian_multi": "BEV",
        "polar_multi": "Polar",
        "range_multi": "Range",
        "fv_multi": "Front"
    }
    # --- END OF MODIFICATION ---

    for dataset_name, cfg in DATASETS.items():
        if not cfg["use"]:
            continue

        out_dir = f"./representation_analysis/RESULTS/media/pr_curves_multi/{dataset_name}"
        os.makedirs(out_dir, exist_ok=True)

        for seq in cfg["seqs"]:

            # Skip NCLT DB seq for PR
            if dataset_name == "NCLT" and seq == cfg["seqs"][0]:
                continue

            plt.figure(figsize=(7,6))
            plotted_any = False

            for rep in REPRESENTATIONS:
                if rep not in PR_DATA[dataset_name][seq]:
                    continue

                P, R, r1 = PR_DATA[dataset_name][seq][rep]
                
                # --- START OF MODIFICATION ---
                # 2. Use the mapping to get the new label, defaulting to the original 'rep' if not found
                new_label = LEGEND_MAPPING.get(rep, rep) 
                
                # Use the new label in the plot call
                plt.plot(R, P, 
                         label=f"{new_label} (R@1={r1:.2f})", linestyle='-')
                         #marker='|',             # Use '|' for the vertical line marker
                         
               
                
                plotted_any = True

     
            if plotted_any:
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"{dataset_name} – {seq}")
                #plt.legend()
                plt.grid(True)

                save_path = os.path.join(out_dir, f"{seq}.png")
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"Saved: {save_path}")

            else:
                plt.close()
                print(f"Skipped seq {seq}: no PR data")




'''
# Optimized and cleaned version of your evaluation script.
# Imported functions are assumed correct and untouched.

import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import isfile, join

import dataset_manager.kitti_dataset as kitti_dataset
import dataset_manager.nclt_dataset as nclt_dataset
import dataset_manager.helilpr_dataset as helilpr_dataset
import dataset_manager.toyota_dataset as toyota_dataset
from models.model_selector import model_selector

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def infer(model, eval_set, device, batch_size, num_workers, return_local_feats=False):
    loader = DataLoader(eval_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()
    model.to(device)

    total_time, batches = 0.0, 0
    all_global, all_local = [], []

    with torch.no_grad():
        for imgs, _ in tqdm(loader, disable=True):
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            local_feat, global_desc = model(imgs)
            #if device.type == 'cuda': torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            batches += 1

            all_global.append(global_desc.cpu().numpy())
            if return_local_feats:
                all_local.append(local_feat.cpu().numpy())

    print(f"Avg inference time per batch: {total_time / max(batches,1):.4f}s")

    global_concat = np.concatenate(all_global, axis=0)
    if return_local_feats:
        return np.concatenate(all_local, axis=0), global_concat
    return global_concat


def load_weights(path, model):
    ckpt_path = os.path.join(path, 'model_best.pth.tar')
    if not os.path.isfile(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}")
        return model
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model

# === PR Curve Plotting Utilities ===


def plot_pr_curve(precisions, recalls, title, save_path=None):
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    #  Save the figure if a path is specified
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Plot successfully saved to: {save_path}")
    else:
        # Note: If no save_path is provided, the plot is created but not saved.
        # It is still closed below.
        print("Warning: No save_path provided, figure will be created but not saved.")
    plt.close()
    print('I should have plotted')


if __name__ == "__main__":

    device = torch.device("cuda:1")

    REPRESENTATIONS = ['cartesian_multi','polar_multi'] #'range_multi','fv_multi']
    DATASETS = {
        "KITTI": {
            "use": False,
            "seqs": ['02', '05', '06'],
            "loader": lambda seq, rep: kitti_dataset.InferDataset(seq=seq, representation=rep),
            "eval": kitti_dataset.evaluateResults_pr_only
        },
        "NCLT": {
            "use": True,
            "seqs": ['2012-01-15','2012-02-04','2012-06-15','2013-02-23'],
            "loader": lambda seq, rep: nclt_dataset.InferDataset(seq=seq, representation=rep),
            "eval": nclt_dataset.evaluateResults_pr_only
        },
        "HELILPR": {
            "use": False,
            "seqs": ['rb01','rb02','rb03'],
            "loader": lambda seq, rep: helilpr_dataset.InferDataset(seq=seq, representation=rep),
            "eval": helilpr_dataset.evaluateResults_pr_only
        },
        "TOYOTA": {
            "use": False,
            "seqs": ['00'],
            "loader": lambda seq, rep: toyota_dataset.InferDataset(seq=seq, representation=rep),
            "eval": toyota_dataset.evaluateResults_pr_only
        }
    }

    # Reproducibility
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed(seed)

    MODEL = "dino3vlad"
    model, proj_dim = model_selector(MODEL, device)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    
    for rep in REPRESENTATIONS:
        print(f"\n========== Representation: {rep} ==========")
        model = load_weights(f"./representation_analysis/RESULTS/Trainings/{rep}/checkpoint/", model)

        for dataset_name, cfg in DATASETS.items():
            if not cfg["use"]:
                continue

            print(f"\n--- Dataset: {dataset_name} ---")
            precisions_all = []
            recalls_all = []
   

            for seq in cfg["seqs"]:
                print(f"Processing seq {seq}")
                test_set = cfg["loader"](seq, rep)
                global_descs = infer(model, test_set, device, batch_size=32, num_workers=4)

                # Special handling for NCLT: first sequence = DB, others = query
                if dataset_name == "NCLT":
 
                # Build DB once: first seq
                    if seq == cfg["seqs"][0]:
                        nclt_db_descs = global_descs
                        nclt_db_set = test_set
                        print(f"Using {seq} as NCLT database")
                        continue
         
                    # Query against DB
                    descriptors = [nclt_db_descs, global_descs]   
                    dataset     = [nclt_db_set,   test_set]       

                    results = cfg["eval"](
                        seq,
                        descriptors, 
                        dataset
                    )

                    del dataset
                    del descriptors

                else:
                    results = cfg["eval"](seq, global_descs, test_set)

                # evaluation API varies between datasets
                #results = cfg["eval"](seq, global_descs, test_set)

                recall_at1 = results[0]
                precisions_sorted = results[1] if len(results) > 1 else None
                recalls_sorted = results[2] if len(results) > 2 else None

                print(f"Seq {seq} Recall@1 = {recall_at1:.2f}")
                # --- PR curve plotting ---
                if precisions_sorted is not None and recalls_sorted is not None:
                    out_dir = f"./representation_analysis/RESULTS/media/pr_curves/{rep}/{dataset_name}"
                    os.makedirs(out_dir, exist_ok=True)
                    save_path = os.path.join(out_dir, f"{seq}.png")
                    plot_pr_curve(precisions_sorted, recalls_sorted, f"{rep} - {dataset_name} - {seq} ({recall_at1:.2f})", save_path)
                if precisions_sorted is not None:
                    precisions_all.append(precisions_sorted)
                if recalls_sorted is not None:
                    recalls_all.append(recalls_sorted)
'''
