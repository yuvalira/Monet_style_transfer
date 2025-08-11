import os
import tensorflow as tf
from typing import List, Tuple, Optional

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = [256, 256]

_TFREC_FEATURES = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.string),
}

def _decode_image(image_bytes: tf.Tensor) -> tf.Tensor:
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0  # [-1, 1]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def _parse_return_name(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES)
    img = _decode_image(ex["image"])
    name = ex["image_name"]
    return img, name

def _parse_image_only(example: tf.Tensor) -> tf.Tensor:
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES)
    return _decode_image(ex["image"])

def _glob_domain_files(gcs_path: str) -> Tuple[List[str], List[str]]:
    monet = tf.io.gfile.glob(f"{gcs_path}/monet_tfrec/*.tfrec")
    photo = tf.io.gfile.glob(f"{gcs_path}/photo_tfrec/*.tfrec")
    return monet, photo

def _collect_names(tfrec_files: List[str], limit: Optional[int] = None) -> List[bytes]:
    names: List[bytes] = []
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).map(
        lambda e: tf.io.parse_single_example(e, _TFREC_FEATURES)["image_name"],
        num_parallel_calls=AUTOTUNE,
    )
    for i, n in enumerate(ds.as_numpy_iterator()):
        names.append(n)
        if limit is not None and i + 1 >= limit:
            break
    return names

def create_manifests(
    gcs_path: str,
    out_dir: str = "data/splits",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    tf.io.gfile.makedirs(out_dir)
    monet_files, photo_files = _glob_domain_files(gcs_path)

    monet_names = _collect_names(monet_files)
    photo_names = _collect_names(photo_files)

    # deterministic shuffle
    monet_names = tf.random.shuffle(monet_names, seed=seed).numpy().tolist()
    photo_names = tf.random.shuffle(photo_names, seed=seed + 1).numpy().tolist()

    def _split(names: List[bytes]) -> Tuple[List[bytes], List[bytes]]:
        n_train = int(len(names) * train_ratio)
        return names[:n_train], names[n_train:]

    monet_tr, monet_te = _split(monet_names)
    photo_tr, photo_te = _split(photo_names)

    def _write(path: str, arr: List[bytes]) -> None:
        with tf.io.gfile.GFile(path, "w") as f:
            for b in arr:
                f.write(b.decode("utf-8") + "\n")

    _write(os.path.join(out_dir, "train_monet.txt"), monet_tr)
    _write(os.path.join(out_dir, "test_monet.txt"), monet_te)
    _write(os.path.join(out_dir, "train_photo.txt"), photo_tr)
    _write(os.path.join(out_dir, "test_photo.txt"), photo_te)

def _load_manifest(path: str) -> tf.lookup.StaticHashTable:
    with tf.io.gfile.GFile(path, "r") as f:
        keys = [line.strip() for line in f if line.strip()]
    keys_tf = tf.constant(keys, dtype=tf.string)
    vals_tf = tf.ones_like(keys_tf, dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tf, vals_tf), default_value=0
    )
    return table

def load_split(
    gcs_path: str,
    domain: str,            # "photo" or "monet"
    split: str,             # "train" or "test"
    batch_size: int = 1,
    shuffle: bool = False,
    repeat: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    assert domain in ("photo", "monet")
    assert split in ("train", "test")

    monet_files, photo_files = _glob_domain_files(gcs_path)
    files = monet_files if domain == "monet" else photo_files

    manifest_path = f"data/splits/{split}_{domain}.txt"
    table = _load_manifest(manifest_path)

    def _keep(img: tf.Tensor, name: tf.Tensor) -> tf.Tensor:
        return tf.greater(table.lookup(name), 0)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_return_name, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(_keep)
    ds = ds.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
