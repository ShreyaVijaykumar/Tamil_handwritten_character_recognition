# PS-39: Handwritten Tamil Character Recognition (HPL IWFHR-06, offline)

**Register No.:** RA2311026050034

Classification of isolated offline (bi-level TIFF) handwritten Tamil characters into 156 classes.
Specific challenge: characters that differ only by a small stroke (confusable classes), plus writer leakage.

## Dataset
HP Labs India Isolated Handwritten Tamil Character Dataset (IWFHR 2006 competition), offline version:
<https://lipitk.sourceforge.net/datasets/tamilchardata.htm>

The dataset is **not** included in this repository. Download it and place it in the project folder as:

```
dataset/
  train/hpl-tamil-iwfhr06-train-offline/usr_*/CCCtNN.tiff
  test/hpl-tamil-iwfhr06-test-offline/NNNNN.tiff
  hpl-tamil-iwfhr06-test-groundtruth.txt
```

The full set was used (no subset): 50,683 training images from 117 writers, 26,926 official test images, 156 classes.

## Method
- **Pre-processing:** ink mask, crop to bounding box, pad to square, resize to 64x64, scale to [0,1].
- **Split:** writer-disjoint train/validation split of the training writers (seed 42): 42,202 train images (99 writers) and 8,481 validation images (18 writers). The official test set is used once, at the end.
- **Models:** a custom 4-stage CNN trained from scratch, and an ImageNet-pretrained ResNet-18 (fully fine-tuned, single-channel input). Stroke-preserving augmentation (small affine changes and stroke dilation; no erasing or erosion). A class-weighted loss was tried for class imbalance.
- **Novelty tried:** CGER (Confusion-Guided Expert Refinement), expert networks for confusion groups mined from the validation confusion matrix. It did **not** improve the test result (negative result, analysed in the report).

## Results (official test set, evaluated once)
| Model | Accuracy (%) | Macro F1 (%) | Top-5 (%) | Errors |
|---|---|---|---|---|
| ResNet-18 (final model) | 93.67 | 93.25 | 99.61 | 1,704 |
| ResNet-18 + CGER | 93.40 | 92.98 | - | 1,778 |

Validation accuracy (8,481 images, used for model selection): CNN 92.89, CNN + augmentation 93.48, CNN + augmentation + class weights 93.04, ResNet-18 93.59.

Errors are concentrated in parallel rows of classes (76-80, 122-126, 127-131) and in the pair 8 / 32. The analysis is in the report.

## Files
- `RA2311026050034_PS39.ipynb` - the full pipeline with outputs: data loading, class counts, sample images, splits, training, evaluation, error analysis, CGER and report figures.
- `train_split.csv`, `validation_split.csv` - the fixed writer-disjoint splits. The `path` column holds paths from the author's machine; the notebook regenerates them when run.
- `report_figs/` - the figures used in the report, with the top-confused-pairs table.
- `requirements.txt` - library versions.

The notebook creates an `outputs/` folder (trained weights, cached image arrays, experiment tables) when it is run. Those files are not in the repository.

## How to run
1. Install the requirements (`pip install -r requirements.txt`). For GPU training, install the CUDA build of PyTorch from <https://pytorch.org> instead of the default build. The reported run used CUDA 12.6.
2. Place the dataset as shown above.
3. Open the notebook from the project folder (or set the environment variable `PS39_ROOT` to the project folder) and run all cells. A GPU is recommended.

## Reproducibility
Seed 42 is set for Python, NumPy and PyTorch. The validation split uses `GroupShuffleSplit(random_state=42)` over writers and is saved in the two CSV files. The reported run used Python 3.11, torch 2.14.1, torchvision 0.29.1, numpy 2.4.6, pandas 3.1.1 and scikit-learn 1.9.1. GPU training is not bit-for-bit deterministic, so a re-run may differ slightly.

## Academic integrity
- **AI assistance:** Claude (Anthropic) was used to help write the code and the report draft. All experiments were run and checked by the author.
- **Prior work consulted:** Vijayaraghavan and Sra, "Handwritten Tamil Recognition using a Convolutional Neural Network"; the uTHCD benchmark paper; Ramakrishnan and Sundaram on confused online Tamil characters. ResNet-18 ImageNet weights are from torchvision.
- **Dataset:** HP Labs India, Isolated Handwritten Tamil Character Dataset (IWFHR 2006).
