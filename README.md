# Winter RSC Classification with SSL + Meta-Learning

This project implements a reproducible Python research pipeline for 5-class winter Road Surface Condition (RSC) classification using self-supervised learning (SSL), class-balanced supervised fine-tuning, episodic meta-learning, hard rare-class episodes, representation analysis, calibration analysis, and Grad-CAM / attention-style interpretation.

The proposed method is **not purely few-shot, but few-shot-aware and rare-class-aware**. The current 5-class dataset is not small overall, but `3 One Track - Partly` is highly underrepresented and visually subtle. The pipeline therefore optimizes macro-F1, balanced accuracy, and One Track recall/F1 rather than only overall accuracy.

## Research Motivation

Winter RSC classes can differ by fine-grained visual evidence: tire paths, lane center snow, shoulder snow, road texture, and partially covered asphalt. The rare class `3 One Track - Partly` is especially difficult because it can be confused with:

- `1 Centre - Partly`
- `2 Two Track - Partly`
- `4 Fully`

The framework combines:

1. SSL pretraining on unlabeled winter-road imagery.
2. RSC-preserving augmentations.
3. Class-balanced supervised fine-tuning.
4. Balanced episodic learning.
5. Hard episodes focused on visually confusing class pairs.
6. Future few-shot adaptation to a sixth class, `5 Black Ice`.

## Dataset Structure

Default paths:

```text
Warm-up Dataset/
Dataset_classes/1 Defined/
Output/
```

Required labeled structure:

```text
Dataset_classes/
└── 1 Defined/
    ├── train/
    │   ├── 0 Bare/
    │   ├── 1 Centre - Partly/
    │   ├── 2 Two Track - Partly/
    │   ├── 3 One Track - Partly/
    │   └── 4 Fully/
    └── val/
        ├── 0 Bare/
        ├── 1 Centre - Partly/
        ├── 2 Two Track - Partly/
        ├── 3 One Track - Partly/
        └── 4 Fully/
```

The code validates every required split/class folder before training and raises a specific error if any folder is missing.

Known approximate class distribution:

| Class | Name | Samples |
|---:|---|---:|
| 0 | Bare | 3335 |
| 1 | Centre - Partly | 1505 |
| 2 | Two Track - Partly | 2191 |
| 3 | One Track - Partly | 434 |
| 4 | Fully | 3139 |

## Learning Pipeline

The full method follows five steps:

1. Build the Warm-Up SSL dataset from unlabeled winter-road images.
2. Train an SSL encoder with contrastive learning.
3. Use RSC-preserving augmentations that retain tire tracks and road-surface evidence.
4. Fine-tune on the 5-class labeled dataset with class-balanced loss/sampling.
5. Continue with episodic meta-learning and hard episodes focused on One Track sensitivity.

## Models

Both models run sequentially by default:

- **ConvNeXt** via `timm`, default `convnext_base_in22k`; supports `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_base_in22k`, `convnext_large`.
- **DINO / DINOv2-style ViT** via `timm`, default `vit_base_patch14_dinov2.lvd142m`.

If the requested ConvNeXt or DINO model is unavailable in the local `timm` version, the code logs the issue and falls back to `convnext_base` or `vit_base_patch16_224`, respectively.

ConvNeXt uses Grad-CAM on the final convolutional feature layer. DINO/ViT uses transformer-compatible visualization where available; when attention tensors are not exposed by the installed backbone, the code generates explicitly labeled input-gradient saliency instead of silently faking Grad-CAM.

## Experiments

The first experiment is:

### E1: SSL + Prototypical Networks

Key: `SSL_Prototypical`

Purpose: test whether SSL-pretrained winter-road embeddings support balanced 5-way K-shot classification. The code saves prototype vectors, prototype distance matrices, support/query indices, query metrics, and query predictions. Defaults are `support_per_class=60`, `query_per_class=60`, and Euclidean prototype distance.

### E2: SSL + Hard Prototypical Episodes

Key: `SSL_Hard_Prototypical`

Purpose: increase sensitivity to `3 One Track - Partly` using hard episodes. At least 50% of episodes emphasize hard relationships involving One Track and other partly/fully snow-covered classes.

### E3: SSL + Class-Balanced Supervised Fine-Tuning

Key: `SSL_ClassBalanced_FineTune`

Purpose: evaluate SSL domain pretraining plus imbalance-aware supervised learning. Supports `ce`, `weighted_ce`, and `focal`, plus `standard` or `balanced` sampling. Defaults are `weighted_ce` and `balanced`.

### E4: SSL + MAML / ANIL

Key: `SSL_MAML_ANIL`

Purpose: evaluate optimization-based episodic adaptation after SSL. ANIL adapts the classifier head by default. MAML supports `head`, `last_block`, and `full` adaptation scopes using a first-order practical implementation suitable for large backbones.

### E5: SSL + Supervised Contrastive / Metric Learning

Key: `SSL_MetricLearning`

Purpose: improve the embedding space for fine-grained RSC classes. Supports supervised contrastive loss and triplet loss with hard-negative emphasis.

### E6: SSL + Hybrid Fine-Tune + Episodic Meta-Learning

Key: `SSL_Hybrid_FineTune_Episodic`

This is the recommended current-task method:

```text
Warm-Up SSL
→ class-balanced supervised fine-tuning
→ balanced prototypical episodes
→ hard One Track episodes
→ final 5-class evaluation
```

It saves phase checkpoints:

- `checkpoints/ssl_encoder.pt`
- `checkpoints/supervised_finetuned.pt`
- `checkpoints/episodic_balanced.pt`
- `checkpoints/episodic_hard_final.pt`
- `checkpoints/best_model.pt`

### E7: Simulated Future-Class Few-Shot Adaptation

Key: `SSL_Simulated_FutureClass`

Purpose: simulate future black-ice adaptation before black-ice data exists. The default pseudo-novel class is `3 One Track - Partly`, evaluated at 1, 5, 10, 20, and 40 shots.

## RSC-Preserving Augmentations

Augmentations are defined in `src/augmentations/rsc_augmentations.py` with `light`, `medium`, and `strong` levels.

Allowed transformations are intentionally moderate: brightness/contrast changes, mild color jitter, mild blur, JPEG compression, road-preserving random resized crop, horizontal flip, and small rotations. The pipeline avoids vertical flips, aggressive crops, random erasing over tire tracks, excessive blur, unrealistic color shifts, and synthetic snow that changes the label.

This matters because RSC labels are often determined by subtle tire-track and snow-cover evidence.

## Output Structure

Each experiment/model pair is isolated:

```text
Output/
├── SSL_Prototypical/
│   ├── convnext/
│   └── dino/
├── SSL_Hard_Prototypical/
├── SSL_ClassBalanced_FineTune/
├── SSL_MAML_ANIL/
├── SSL_MetricLearning/
├── SSL_Hybrid_FineTune_Episodic/
├── SSL_Simulated_FutureClass/
└── BlackIce/
```

Inside each model-specific folder:

```text
checkpoints/
logs/
metrics/
plots/
confusion_matrices/
calibration/
confidence/
embeddings/
gradcam/
predictions/
configs/
epoch_progress/
ablations/
reports/
```

## Metrics and Analysis

Every experiment saves:

- accuracy, balanced accuracy, macro-F1, weighted F1
- macro precision/recall
- per-class precision/recall/F1/support
- top-2 accuracy and top-1/top-2 confidence margin
- confusion matrices and normalized confusion matrices
- Expected Calibration Error, Maximum Calibration Error, Brier score, and NLL
- confidence histograms and confidence/error analysis
- UMAP/t-SNE/PCA embedding analysis
- class centroid and intra/inter-class distances
- One Track precision/recall/F1, false-negative rate, false-positive rate, common confusions, and false-negative paths
- episodic support/query metrics where applicable
- few-shot curves where applicable

Epoch-level CSVs are stored under `epoch_progress/`:

- `ssl_epoch_metrics.csv`
- `supervised_epoch_metrics.csv`
- `meta_epoch_metrics.csv`
- `maml_anil_epoch_metrics.csv`
- `learning_rate_schedule.csv`
- `gpu_memory_log.csv`
- `epoch_time_log.csv`

## Grad-CAM / Attention Format

Interpretability panels are saved as PNG files with exactly three panels:

```text
[Original Image]    [Heatmap Only]    [Heatmap Overlaid on Original Image]
```

Metadata below each panel includes:

- experiment
- model
- true class
- predicted class
- top confidence
- probability for each class
- visualization method

ConvNeXt panels use Grad-CAM. DINO/ViT panels are attention visualizations when attention is available, otherwise explicitly labeled input-gradient saliency.

## Commands

Run everything:

```bash
python run_ssl_meta_rsc.py
```

Run all experiments with both models:

```bash
python run_ssl_meta_rsc.py --models convnext dino --experiments all
```

Run only ConvNeXt:

```bash
python run_ssl_meta_rsc.py --models convnext --experiments all
```

Run only DINO:

```bash
python run_ssl_meta_rsc.py --models dino --experiments all
```

Run E1:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Prototypical --models convnext dino
```

Run E2:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Hard_Prototypical --models convnext dino
```

Run E3:

```bash
python run_ssl_meta_rsc.py --experiment SSL_ClassBalanced_FineTune --models convnext dino
```

Run E4:

```bash
python run_ssl_meta_rsc.py --experiment SSL_MAML_ANIL --models convnext dino --meta_algorithm anil
```

Run E5:

```bash
python run_ssl_meta_rsc.py --experiment SSL_MetricLearning --models convnext dino --metric_loss supcon
```

Run E6:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Hybrid_FineTune_Episodic --models convnext dino
```

Run E7:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Simulated_FutureClass --models convnext dino --pseudo_novel_class "3 One Track - Partly"
```

Change support/query samples:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Prototypical --support_per_class 20 --query_per_class 20
```

Run with default support/query explicitly:

```bash
python run_ssl_meta_rsc.py --experiment SSL_Prototypical --support_per_class 60 --query_per_class 60
```

Run ANIL:

```bash
python run_ssl_meta_rsc.py --experiment SSL_MAML_ANIL --meta_algorithm anil --adapt_scope head
```

Run full MAML:

```bash
python run_ssl_meta_rsc.py --experiment SSL_MAML_ANIL --meta_algorithm maml --adapt_scope last_block
```

Run focal loss:

```bash
python run_ssl_meta_rsc.py --experiment SSL_ClassBalanced_FineTune --loss focal --sampler balanced
```

Run supervised contrastive loss:

```bash
python run_ssl_meta_rsc.py --experiment SSL_MetricLearning --metric_loss supcon --hard_negative_mining true
```

## Black-Ice Future Adaptation

Future black-ice data is expected under:

```text
Black-ice/
├── image_1.jpg
├── image_2.jpg
└── ...
```

Black ice may be visually similar to wet pavement, glare, dark asphalt, shadows, or bare pavement. Therefore, image-only black-ice classification should be considered preliminary. The code is designed so future auxiliary context can be added, including temperature, precipitation, road-surface temperature, friction observations, weather history, and glare indicators.

Adapt an existing checkpoint:

```bash
python run_black_ice_pipeline.py --blackice_mode adapt_existing --experiment SSL_Prototypical --models convnext dino --blackice_dir "Black-ice"
```

Train from start:

```bash
python run_black_ice_pipeline.py --blackice_mode train_from_start --experiment SSL_Hybrid_FineTune_Episodic --models convnext dino --blackice_dir "Black-ice"
```

ConvNeXt only:

```bash
python run_black_ice_pipeline.py --blackice_mode adapt_existing --experiment SSL_Prototypical --models convnext --blackice_dir "Black-ice"
```

DINO only:

```bash
python run_black_ice_pipeline.py --blackice_mode adapt_existing --experiment SSL_Prototypical --models dino --blackice_dir "Black-ice"
```

Custom black-ice shot counts:

```bash
python run_black_ice_pipeline.py --blackice_mode adapt_existing --experiment SSL_Prototypical --blackice_shots 5 10 20 40
```

Black-ice outputs are saved under:

```text
Output/BlackIce/<experiment>/<model>/
```

They include 6-class confusion matrices, normalized confusion matrices, black-ice precision/recall/F1, black-ice false-negative rate, calibration plots, confidence histograms, Grad-CAM/attention visualizations, embedding plots, and few-shot adaptation curves.

## Reproducibility

Every run saves:

- `configs/args.json`
- `configs/config.yaml`
- `configs/class_mapping.json`
- `configs/dataset_summary.csv`
- `configs/train_distribution.csv`
- `configs/val_distribution.csv`
- `configs/git_commit.txt`
- checkpoints
- logs
- prediction CSVs
- epoch-level CSVs

The code sets Python, NumPy, and PyTorch seeds and enables deterministic PyTorch behavior where practical.

## Troubleshooting

- If a folder is missing, the runner stops before training and prints the exact missing path.
- If CUDA is requested but unavailable, the runner falls back to CPU and logs the change.
- If the selected ConvNeXt/DINO model is unavailable, the runner falls back to `convnext_base` / `vit_base_patch16_224`.
- If Grad-CAM cannot be applied to a selected transformer, the visualization is labeled as saliency rather than Grad-CAM.
- Defaults use one epoch per phase so commands are executable immediately. Increase `--epochs_ssl`, `--epochs_finetune`, `--epochs_meta`, and `--episodes_per_epoch` for full research runs.

## Recommended Interpretation

Prefer experiments that improve:

1. `3 One Track - Partly` recall and F1.
2. macro-F1 and balanced accuracy.
3. calibration quality, especially ECE and confidence of errors.
4. separation of partly snow-covered classes in embedding plots.
5. Grad-CAM/attention focus on tire paths and road-surface regions.

High overall accuracy with poor One Track recall should not be treated as a successful rare-class-aware result.
