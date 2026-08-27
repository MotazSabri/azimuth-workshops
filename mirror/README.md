# Asset mirrors

Copies of datasets that workshops depend on, served from this repository so a
workshop does not die the day someone else's URL changes.

A mirror is only useful if it is byte-identical to what the `sha256` in the
workshop's `assets:` block pins. Do not "clean up" a file here — if the
upstream copy has a stray header row or Windows line endings, so must this one,
or the hash check will reject the mirror and the fallback will never fire.

To add one:

```
curl -L <upstream-url> -o mirror/<name>
python tools/pin-asset.py <slug>     # if not already pinned
sha256sum mirror/<name>              # must equal the pinned value
```

`ecg5000.csv` is ~7.9 MB, which is fine to keep in git. Anything much larger
belongs in a release asset rather than the tree.

## Dataset citations

### ECG5000

The `ecg5000.csv` mirror is sourced from the ECG dataset distributed by
TensorFlow:

* TensorFlow dataset URL:
  http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv

### CoVoST2 English–Arabic

The English–Arabic translation pairs are from the **CoVoST2-EN-AR-Text**
dataset by ymoslem, hosted on Hugging Face:

* Dataset:
  https://huggingface.co/datasets/ymoslem/CoVoST2-EN-AR-Text
* Training data:
  https://huggingface.co/datasets/ymoslem/CoVoST2-EN-AR-Text/blob/main/data/train-00000-of-00001.parquet
