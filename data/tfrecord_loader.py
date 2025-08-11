import tensorflow as tf
from typing import Iterable, Tuple, List

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = [256, 256]

_TFREC_FEATURES = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.string),
}

def decode_image(image_bytes: tf.Tensor) -> tf.Tensor:
    """Decode JPEG bytes → float32 tensor in [-1, 1], shape (256, 256, 3)."""
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def parse_example(example_proto: tf.Tensor) -> tf.Tensor:
    """Parse a single TFRecord example and return the image tensor only."""
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    image = decode_image(ex["image"])
    return image  # labels/ids are not needed for CycleGAN

def load_dataset(
    filenames: Iterable[str],
    batch_size: int = 1,
    shuffle: bool = False,
    repeat: bool = False,
    drop_remainder: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data pipeline from TFRecord files."""
    files = list(filenames)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def list_domain_files(gcs_path: str) -> Tuple[List[str], List[str]]:
    """Return TFRecord file lists for Monet (B) and Photo (A) domains."""
    monet = tf.io.gfile.glob(f"{gcs_path}/monet_tfrec/*.tfrec")
    photo = tf.io.gfile.glob(f"{gcs_path}/photo_tfrec/*.tfrec")
    return monet, photo

def build_domain_datasets(
    gcs_path: str,
    batch_size: int = 1,
    shuffle: bool = False,
    repeat: bool = False,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Convenience: build monet_ds and photo_ds in one call."""
    monet_files, photo_files = list_domain_files(gcs_path)
    monet_ds = load_dataset(monet_files, batch_size, shuffle, repeat, False, seed)
    photo_ds = load_dataset(photo_files, batch_size, shuffle, repeat, False, seed)
    return monet_ds, photo_ds

def count_examples(filenames: Iterable[str]) -> int:
    """Count number of examples across TFRecord files (fast scan)."""
    n = 0
    for f in filenames:
        for _ in tf.data.TFRecordDataset([f]):
            n += 1
    return n