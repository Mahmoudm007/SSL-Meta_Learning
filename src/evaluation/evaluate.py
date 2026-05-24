from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.class_mapping import ClassMapping, ONE_TRACK_CLASS_ID
from src.evaluation.calibration import save_calibration_outputs
from src.evaluation.confidence import save_confidence_outputs
from src.evaluation.confusion import save_confusion_outputs
from src.evaluation.embeddings import extract_embeddings, save_embedding_outputs
from src.evaluation.gradcam import generate_interpretability_outputs
from src.evaluation.metrics import classification_metrics, prediction_dataframe
from src.supervised.finetune import evaluate_loader
from src.utils.io import ensure_dir, write_dataframe, write_text
from src.visualization.plots import plot_bar, plot_curve, save_placeholder_figure, save_required_curve_figures
from src.visualization.report_figures import save_error_analysis_figures, save_model_comparison_placeholders


def evaluate_and_save_all(
    model: torch.nn.Module,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
    experiment: str,
    model_key: str,
    eval_result: dict | None = None,
    include_black_ice: bool = False,
) -> pd.DataFrame:
    root = Path(output_root)
    ensure_dir(root)
    if eval_result is None:
        loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        criterion = nn.CrossEntropyLoss()
        eval_result = evaluate_loader(model, loader, criterion, device)
    labels = eval_result["labels"]
    probabilities = eval_result["probabilities"]
    paths = eval_result["paths"]
    metrics = classification_metrics(labels, probabilities, class_mapping)
    summary_frame = pd.DataFrame([metrics["summary"]])
    per_class = metrics["per_class"]
    one_track = metrics["one_track"]
    predictions = prediction_dataframe(paths, labels, probabilities, class_mapping)

    write_dataframe(root / "metrics" / "metrics_summary.csv", summary_frame)
    write_dataframe(root / "metrics" / "per_class_metrics.csv", per_class)
    write_dataframe(root / "metrics" / "one_track_metrics.csv", one_track)
    write_dataframe(root / "metrics" / "all_predictions.csv", predictions)
    _save_prediction_subsets(predictions, root)
    save_confusion_outputs(metrics["confusion_matrix"], class_mapping.class_names, root / "confusion_matrices")
    save_calibration_outputs(probabilities, labels, root / "calibration", len(class_mapping.class_names))
    save_confidence_outputs(predictions, root / "confidence", class_mapping.class_names)
    save_error_analysis_figures(predictions, root)
    _save_top_level_plots(root, per_class)
    save_required_curve_figures(root)
    save_model_comparison_placeholders(root)
    try:
        embeddings, embedding_labels, embedding_paths = extract_embeddings(model, val_dataset, device, args.batch_size, args.num_workers)
        save_embedding_outputs(embeddings, embedding_labels, embedding_paths, class_mapping, root / "embeddings", predictions)
    except Exception as exc:
        logger.error("Embedding analysis failed: %s", exc)
        save_placeholder_figure(root / "embeddings" / "umap_embeddings.png", "UMAP Embeddings", f"Embedding analysis failed: {exc}")
        save_placeholder_figure(root / "embeddings" / "tsne_embeddings.png", "t-SNE Embeddings", f"Embedding analysis failed: {exc}")
        save_placeholder_figure(root / "embeddings" / "pca_embeddings.png", "PCA Embeddings", f"Embedding analysis failed: {exc}")
    if args.generate_gradcam:
        generate_interpretability_outputs(
            model,
            predictions,
            root,
            class_mapping,
            experiment,
            model_key,
            args.image_size,
            device,
            logger,
            max_images=args.max_gradcam_images,
            include_black_ice=include_black_ice,
            primary_subdir="all_validation",
        )
        query_path = root / "predictions" / "query_predictions.csv"
        if query_path.exists() and query_path.stat().st_size > 0:
            try:
                query_predictions = pd.read_csv(query_path)
                if len(query_predictions):
                    generate_interpretability_outputs(
                        model,
                        query_predictions,
                        root,
                        class_mapping,
                        experiment,
                        model_key,
                        args.image_size,
                        device,
                        logger,
                        max_images=args.max_gradcam_images,
                        include_black_ice=include_black_ice,
                        primary_subdir="all_query",
                    )
            except Exception as exc:
                logger.error("Query Grad-CAM/attention generation failed: %s", exc)
    else:
        write_dataframe(root / "gradcam" / "gradcam_metadata.csv", pd.DataFrame([{"note": "Grad-CAM generation disabled by CLI"}]))
    _write_reports(root, experiment, model_key, summary_frame, per_class, one_track, predictions)
    _ensure_optional_metric_files(root)
    logger.info(
        "Final evaluation: accuracy=%.5f macro_f1=%.5f one_track_recall=%.5f",
        float(metrics["summary"].get("overall_accuracy", 0.0)),
        float(metrics["summary"].get("macro_f1", 0.0)),
        float(one_track.iloc[0]["one_track_recall"]) if len(one_track) else 0.0,
    )
    return predictions


def _save_prediction_subsets(predictions: pd.DataFrame, root: Path) -> None:
    write_dataframe(root / "predictions" / "validation_predictions.csv", predictions)
    incorrect = predictions[~predictions["correct"].astype(bool)] if len(predictions) else predictions
    write_dataframe(root / "predictions" / "incorrect_predictions.csv", incorrect)
    write_dataframe(root / "predictions" / "high_confidence_wrong_predictions.csv", incorrect.sort_values("confidence", ascending=False).head(100) if len(incorrect) else incorrect)
    correct = predictions[predictions["correct"].astype(bool)] if len(predictions) else predictions
    write_dataframe(root / "predictions" / "low_confidence_correct_predictions.csv", correct.sort_values("confidence").head(100) if len(correct) else correct)
    one_track_fn = predictions[(predictions["true_label_id"] == ONE_TRACK_CLASS_ID) & (predictions["pred_label_id"] != ONE_TRACK_CLASS_ID)] if len(predictions) else predictions
    write_dataframe(root / "predictions" / "one_track_false_negatives.csv", one_track_fn)


def _save_top_level_plots(root: Path, per_class: pd.DataFrame) -> None:
    plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", ["train_loss", "val_loss"], root / "plots" / "training_curves.png", "Training Curves")
    plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", ["train_loss", "val_loss"], root / "plots" / "loss_curves.png", "Loss Curves")
    plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", ["train_accuracy", "val_accuracy"], root / "plots" / "accuracy_curves.png", "Accuracy Curves")
    plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", ["train_macro_f1", "val_macro_f1"], root / "plots" / "macro_f1_curves.png", "Macro-F1 Curves")
    plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", ["train_balanced_accuracy", "val_balanced_accuracy"], root / "plots" / "balanced_accuracy_curves.png", "Balanced Accuracy Curves")
    plot_bar(per_class, "label_name", "f1", root / "plots" / "class_f1_barplot.png", "Per-Class F1", "F1")
    save_placeholder_figure(root / "plots" / "model_comparison_summary.png", "Model Comparison Summary", "Aggregated after both ConvNeXt and DINO complete.")
    save_placeholder_figure(root / "plots" / "ssl" / "ssl_embedding_umap_before_after.png", "SSL UMAP Before/After", "Final embeddings are saved in embeddings/; before/after comparison requires cached pre-SSL features.")
    save_placeholder_figure(root / "plots" / "ssl" / "ssl_embedding_tsne_before_after.png", "SSL t-SNE Before/After", "Final embeddings are saved in embeddings/; before/after comparison requires cached pre-SSL features.")
    save_placeholder_figure(root / "plots" / "meta" / "prototype_distance_heatmap.png", "Prototype Distance Heatmap", "Prototype distances are saved for prototypical experiments.")


def _write_reports(root: Path, experiment: str, model_key: str, summary: pd.DataFrame, per_class: pd.DataFrame, one_track: pd.DataFrame, predictions: pd.DataFrame) -> None:
    summary_dict = summary.iloc[0].to_dict() if len(summary) else {}
    report = [
        f"# Experiment Summary: {experiment} / {model_key}",
        "",
        "This run evaluates winter RSC classification with a few-shot-aware and rare-class-aware SSL/meta-learning pipeline.",
        "",
        "## Summary Metrics",
    ]
    for key, value in summary_dict.items():
        report.append(f"- {key}: {value}")
    report.extend(["", "## Per-Class Metrics", per_class.to_markdown(index=False) if len(per_class) else "No per-class metrics available."])
    write_text(root / "reports" / "experiment_summary.md", "\n".join(report) + "\n")
    write_text(root / "reports" / "final_results_table.md", (summary.to_markdown(index=False) if len(summary) else "No summary metrics available.") + "\n")
    wrong = predictions[~predictions["correct"].astype(bool)] if len(predictions) else predictions
    write_text(root / "reports" / "failure_case_summary.md", f"# Failure Case Summary\n\nTotal validation errors: {len(wrong)}\n\nTop errors are saved under `predictions/incorrect_predictions.csv`.\n")
    if len(one_track):
        row = one_track.iloc[0].to_dict()
        one_track_fn = predictions[(predictions["true_label_id"] == 3) & (predictions["pred_label_id"] != 3)] if len(predictions) else predictions
        top_paths = "\n".join(f"- {path}" for path in one_track_fn["image_path"].head(20).tolist())
        analysis = f"""# One Track - Partly Analysis

- total One Track validation samples: {row.get('one_track_samples', 0)}
- number of correct predictions: {row.get('one_track_correct', 0)}
- number of false negatives: {row.get('one_track_false_negatives', 0)}
- One Track precision: {row.get('one_track_precision', 0)}
- One Track recall: {row.get('one_track_recall', 0)}
- One Track F1: {row.get('one_track_f1', 0)}
- One Track false-negative rate: {row.get('one_track_false_negative_rate', 0)}
- most common confusion class: {row.get('most_common_confusion_class', 'none')}
- average confidence when correctly classified: {row.get('avg_confidence_correct', 'nan')}
- average confidence when misclassified: {row.get('avg_confidence_misclassified', 'nan')}

## Top 20 False-Negative Image Paths
{top_paths if top_paths else '- none'}

## Interpretation
Use the confusion distribution and Grad-CAM/attention panels to determine whether One Track errors concentrate toward Centre - Partly, Two Track - Partly, or Fully.
"""
    else:
        analysis = "# One Track - Partly Analysis\n\nNo One Track metrics were available.\n"
    write_text(root / "reports" / "one_track_analysis.md", analysis)


def _ensure_optional_metric_files(root: Path) -> None:
    optional_files = {
        root / "metrics" / "fewshot_results.csv": pd.DataFrame(columns=["shot_count", "accuracy", "macro_f1", "pseudo_novel_recall", "pseudo_novel_f1"]),
        root / "metrics" / "query_metrics.csv": pd.DataFrame(),
        root / "metrics" / "episode_metrics.csv": pd.DataFrame(),
    }
    for path, frame in optional_files.items():
        if not path.exists():
            write_dataframe(path, frame)
