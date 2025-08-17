from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image

# ---------------------------
# Export Monet JPGs once from TFRecords
# ---------------------------

_TFREC_FEATURES = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.string),
}

def ensure_monet_jpgs_from_tfrec(tfrec_files: Iterable[str], out_dir: str | Path) -> None:
    """Export Monet TFRecords to uint8 JPGs in out_dir if that folder is empty."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_jpg = any(out_dir.glob("*.jpg")) or any(out_dir.glob("*.jpeg")) or any(out_dir.glob("*.png"))
    if has_jpg:
        print(f"[mifid] Found existing images in {out_dir}; skipping export.")
        return

    files = list(tfrec_files)
    if not files:
        raise FileNotFoundError("No Monet TFRecord files provided.")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    def _parse(e):
        ex = tf.io.parse_single_example(e, _TFREC_FEATURES)
        img = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8 [0..255]
        return img

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE).batch(1)
    for i, img in enumerate(ds):
        arr = img[0].numpy()  # uint8
        Image.fromarray(arr).save(out_dir / f"{i+1}.jpg")
    print(f"[mifid] Exported {i+1} Monet images to {out_dir}")

# ---------------------------
# MiFID
# ---------------------------

def _list_images(folder: str | Path) -> List[str]:
    folder = Path(folder)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    files: List[str] = []
    for e in exts:
        files += [str(p) for p in folder.glob(e)]
    return sorted(files)

def _load_and_preprocess(paths: List[str], target_size=(299, 299), batch: int = 32):
    for i in range(0, len(paths), batch):
        batch_paths = paths[i : i + batch]
        imgs = []
        for p in batch_paths:
            raw = tf.io.read_file(p)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, target_size)
            img = tf.cast(img, tf.float32)  # 0..255
            imgs.append(img)
        x = tf.stack(imgs, axis=0)
        yield preprocess_input(x.numpy())  # -> [-1, 1]

def _get_inception_features(paths: List[str], model: tf.keras.Model, batch: int = 32) -> np.ndarray:
    feats = []
    for x in _load_and_preprocess(paths, batch=batch):
        f = model.predict(x, verbose=0)  # [B, 2048]
        feats.append(f)
    return np.concatenate(feats, axis=0)

def _stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def _fid_score(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Frechet distance using tf.linalg.sqrtm"""
    mu1 = tf.convert_to_tensor(mu1, tf.float64)
    mu2 = tf.convert_to_tensor(mu2, tf.float64)
    s1  = tf.convert_to_tensor(sigma1, tf.float64)
    s2  = tf.convert_to_tensor(sigma2, tf.float64)
    diff = mu1 - mu2
    # numeric stabilization
    eye = tf.eye(s1.shape[0], dtype=tf.float64) * eps
    covmean = tf.linalg.sqrtm((s1 + eye) @ (s2 + eye))
    covmean = tf.math.real(covmean)
    fid = tf.tensordot(diff, diff, axes=1) + tf.linalg.trace(s1 + s2 - 2.0 * covmean)
    return float(fid.numpy())

def _cosine_distance_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    sim = A_n @ B_n.T
    return 1.0 - sim  # distance

def compute_mifid(
    real_dir: str | Path,
    gen_dir: str | Path,
    *,
    epsilon: float = 0.10,
    batch: int = 64,
) -> Dict[str, float]:
    """
    MiFID:
        d_ij = 1 - cos(f_gi, f_rj)
        d = (1/N) * sum_i min_j d_ij
        d_thr = d if d < epsilon else 1
        MiFID = FID / d_thr
    """
    real_paths = _list_images(real_dir)
    gen_paths  = _list_images(gen_dir)
    if not real_paths:
        raise FileNotFoundError(f"No images in real_dir: {real_dir}")
    if not gen_paths:
        raise FileNotFoundError(f"No images in gen_dir: {gen_dir}")

    inc = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    f_real = _get_inception_features(real_paths, inc, batch=batch)
    f_gen  = _get_inception_features(gen_paths,  inc, batch=batch)

    mu_r, sig_r = _stats(f_real)
    mu_g, sig_g = _stats(f_gen)
    fid = _fid_score(mu_r, sig_r, mu_g, sig_g)

    D = _cosine_distance_matrix(f_gen, f_real)
    d = float(D.min(axis=1).mean())
    d_thr = d if d < epsilon else 1.0
    mifid = float(fid / d_thr)

    return {"FID": float(fid), "d": d, "d_thr": float(d_thr), "MiFID": mifid}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Compute MiFID between real and generated folders.")
    p.add_argument("--real_dir", required=True, type=str)
    p.add_argument("--gen_dir",  required=True, type=str)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    out = compute_mifid(args.real_dir, args.gen_dir, epsilon=args.epsilon, batch=args.batch)
    print(json.dumps(out, indent=2))