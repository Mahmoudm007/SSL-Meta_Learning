# Recommended Experiment Settings

These are the five most promising settings for the current winter RSC project. They prioritize `3 One Track - Partly` recall/F1, macro-F1, balanced accuracy, embedding separation between partly covered classes, and reusable checkpoints for future black-ice adaptation.

The commands assume the updated project defaults:

- ConvNeXt default: `convnext_base_in22k`
- image size: `512`
- batch size: `32`
- support/query per class: `60/60`
- models: `convnext dino`

The batch size is set to `32` because 512px SSL with ConvNeXt Base can exceed GPU memory at `64`.

## 1. Best Overall: Hybrid Fine-Tune + Hard Episodic Meta-Learning

This is the strongest recommended current-task run because it combines every useful signal: SSL pretraining, weighted supervised learning, balanced sampling, balanced episodes, and hard One Track episodes.

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_Hybrid_FineTune_Episodic \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --loss weighted_ce \
  --sampler balanced \
  --prototype_distance euclidean \
  --support_per_class 60 \
  --query_per_class 60 \
  --epochs_ssl 20 \
  --epochs_finetune 30 \
  --epochs_meta 20 \
  --episodes_per_epoch 50 \
  --augmentation_strength medium
```

Why this is promising:

- `weighted_ce + balanced` directly handles class imbalance during fine-tuning.
- Prototypical episodes force balanced class treatment regardless of class frequency.
- Hard episodes specifically target One Track vs Centre, Two Track, and Fully.
- Euclidean prototypes are a strong default for compact class clusters.
- This produces the most useful checkpoint for later black-ice adaptation.

Primary metrics to judge:

- One Track recall and F1
- macro-F1
- balanced accuracy
- hard episode One Track recall
- calibration ECE

## 2. Best Embedding Quality: Supervised Contrastive Metric Learning

This setting is designed to improve feature geometry for visually similar RSC classes. It is especially useful when confusion comes from embedding overlap rather than classifier bias.

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_MetricLearning \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --metric_loss supcon \
  --hard_negative_mining true \
  --loss weighted_ce \
  --sampler balanced \
  --epochs_ssl 20 \
  --epochs_meta 25 \
  --epochs_finetune 20 \
  --augmentation_strength medium
```

Why this is promising:

- Supervised contrastive loss pulls same-class winter road embeddings together.
- Hard negatives emphasize subtle confusions such as One Track vs Two Track and Two Track vs Fully.
- The classifier then trains on a better-organized embedding space.
- This should improve UMAP/t-SNE separation and nearest-neighbor retrieval quality.

Primary metrics to judge:

- class centroid distances
- intra-/inter-class distance ratio
- hard-negative pair performance
- One Track recall/F1
- macro-F1

## 3. Best Rare-Class Classifier Baseline: SSL + Focal Fine-Tuning

This is the strongest supervised-only baseline for the rare One Track class. It is simpler than the hybrid method and useful for isolating whether episodic learning is actually adding value.

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_ClassBalanced_FineTune \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --loss focal \
  --sampler balanced \
  --epochs_ssl 20 \
  --epochs_finetune 40 \
  --augmentation_strength medium
```

Why this is promising:

- Focal loss emphasizes difficult and misclassified samples.
- Balanced sampling reduces majority-class dominance.
- It directly tests whether One Track errors are mostly due to supervised imbalance.
- It is easier to compare against classical non-meta training.

Primary metrics to judge:

- One Track false-negative rate
- One Track confidence when wrong
- macro recall
- macro-F1
- high-confidence wrong predictions

## 4. Best Rare-Class Episodic Run: SSL + Hard Prototypical Episodes

This setting focuses the episodic sampler on visually confusing class relationships, with at least 50% hard episodes and a stronger recommended hard episode probability.

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_Hard_Prototypical \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --prototype_distance euclidean \
  --support_per_class 60 \
  --query_per_class 60 \
  --hard_episode_probability 0.70 \
  --epochs_ssl 20 \
  --epochs_meta 30 \
  --episodes_per_epoch 60 \
  --augmentation_strength medium
```

Why this is promising:

- Episodes are class-balanced by construction.
- Hard episodes repeatedly expose the model to One Track confusions.
- Prototype distance matrices reveal whether One Track is too close to Centre, Two Track, or Fully.
- False-negative lists and partly-class confusion matrices directly support failure analysis.

Primary metrics to judge:

- hard episode accuracy
- hard episode One Track recall
- prototype distances involving One Track
- partly-class confusion matrix
- One Track false negatives

## 5. Best Stable Meta-Adaptation: SSL + ANIL

This is the most stable optimization-based meta-learning setting for large ConvNeXt/DINO backbones. ANIL adapts only the classifier head, which reduces overfitting and memory pressure compared with full MAML.

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_MAML_ANIL \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --meta_algorithm anil \
  --adapt_scope head \
  --inner_steps 5 \
  --inner_lr 1e-3 \
  --outer_lr 1e-4 \
  --support_per_class 60 \
  --query_per_class 60 \
  --epochs_ssl 20 \
  --epochs_meta 30 \
  --episodes_per_epoch 50 \
  --augmentation_strength medium
```

Why this is promising:

- ANIL is usually more stable than full MAML for large image encoders.
- It tests whether the SSL encoder already contains adaptable RSC features.
- It provides pre-/post-adaptation query accuracy and One Track recall.
- It is a useful bridge toward future black-ice few-shot adaptation.

Primary metrics to judge:

- pre- vs post-adaptation query accuracy
- adaptation gain
- One Track post-adaptation recall
- macro-F1 after each inner step
- inner-loop loss stability

## Optional Swap-In for Future Black-Ice Readiness

If future-class adaptation is the main priority, run this in addition to the top five or swap it for the ANIL run:

```bash
python run_ssl_meta_rsc.py \
  --experiment SSL_Simulated_FutureClass \
  --models convnext dino \
  --convnext_name convnext_base_in22k \
  --pseudo_novel_class "3 One Track - Partly" \
  --fewshot_values 1 5 10 20 40 60 \
  --prototype_distance euclidean \
  --epochs_ssl 20 \
  --support_per_class 60 \
  --query_per_class 60 \
  --augmentation_strength medium
```

Use this to estimate whether the trained representation can absorb a rare/new class with limited support samples before actual black-ice labels are available.
