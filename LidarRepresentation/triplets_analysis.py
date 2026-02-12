import os, cv2, numpy as np
from itertools import combinations
import mkl, cv2

# Limit internal threading for OpenCV and NumPy
try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

BASE = "/dataset/kitti-dataset/channel_analysis/sequences/00"
CHANNELS = [
    "omnivariance","anisotropy","planarity","linearity",
    "eigenentropy","sphericity","albedo","intensity",
    "density","curvature","height_norm"
]
COMBS = list(combinations(CHANNELS, 3))

def load_gray(channel, idx):
    path = f"{BASE}/{channel}/{idx:06d}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255 if img is not None else None

def sqrt_volume(a, b, c):
    U = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    return np.sqrt(max(np.linalg.det(U.T @ U), 0.0))

if __name__ == "__main__":
    n_images = len(os.listdir(os.path.join(BASE, CHANNELS[0])))
    results = np.zeros((n_images, len(COMBS)), dtype=np.float32)

    for i in range(n_images):
        vols = []
        for comb in COMBS:
            imgs = [load_gray(ch, i) for ch in comb]
            if any(x is None for x in imgs):
                vols.append(np.nan)
                continue
            vols.append(sqrt_volume(*imgs))
        results[i] = vols
        if i % 100 == 0:
            print(f"Processed {i}/{n_images}")

    np.save("informative_volumes.npy", results)
    print("Saved:", results.shape)
