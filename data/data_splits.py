# deterministic_photo_split.py
import tensorflow as tf
from typing import Tuple

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (256, 256)

_TFREC_FEATURES = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.string),
}

def _decode_image(image_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 127.5 - 1.0   # [-1, 1]
    return img

def _parse_return_name(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES)
    img = _decode_image(ex["image"])
    name = ex["image_name"]
    return img, name

def _photo_files(gcs_path: str):
    return tf.io.gfile.glob(f"{gcs_path}/photo_tfrec/*.tfrec")

def _is_in_split(name: tf.Tensor, split: tf.Tensor, train_ratio: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """
    Deterministic split based on hashing(image_name || seed).
    No external files; identical across notebooks when seed/train_ratio match.
    """
    salted = tf.strings.join([name, tf.strings.as_string(seed)], separator=":")
    bucket = tf.strings.to_hash_bucket_fast(salted, 10_000)
    cutoff = tf.cast(tf.math.round(train_ratio * 10_000.0), tf.int64)
    is_train = bucket < cutoff
    return tf.where(tf.equal(split, "train"), is_train, tf.logical_not(is_train))

def load_photo_split(
    gcs_path: str,
    split: str,                 # "train" or "test"
    train_ratio: float = 0.9,
    seed: int = 42,
    batch_size: int = 1,
    shuffle: bool = False,
    repeat: bool = False,
) -> tf.data.Dataset:
    assert split in ("train", "test")
    files = _photo_files(gcs_path)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_return_name, num_parallel_calls=AUTOTUNE)

    # filter by deterministic split
    ds = ds.filter(lambda img, name: _is_in_split(
        name,
        tf.constant(split),
        tf.constant(train_ratio, tf.float32),
        tf.constant(seed, tf.int64))
    )

    # drop names after filtering
    ds = ds.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# Counters for sanity checking identical splits in both notebooks
def count_examples(gcs_path: str, train_ratio: float = 0.9, seed: int = 42) -> Tuple[int, int]:
    files = _photo_files(gcs_path)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).map(_parse_return_name, num_parallel_calls=AUTOTUNE)

    def _to_flags(name):
        salted = tf.strings.join([name, tf.strings.as_string(seed)], separator=":")
        bucket = tf.strings.to_hash_bucket_fast(salted, 10_000)
        cutoff = tf.cast(tf.math.round(train_ratio * 10_000.0), tf.int64)
        return bucket < cutoff

    flags = ds.map(lambda img, name: _to_flags(name), num_parallel_calls=AUTOTUNE)
    n_train = sum(int(x) for x in flags.as_numpy_iterator())
    total = sum(1 for _ in tf.data.TFRecordDataset(files))
    n_test = total - n_train
    return n_train, n_test
