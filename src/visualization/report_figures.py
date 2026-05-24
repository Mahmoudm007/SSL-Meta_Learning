from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir
from src.visualization.galleries import save_error_gallery
from src.visualization.plots import plot_bar, save_placeholder_figure


def save_error_analysis_figures(predictions: pd.DataFrame, output_root: str | Path) -> None:
    root = Path(output_root)
    errors_dir = ensure_dir(root / "plots" / "errors")
    wrong = predictions[~predictions["correct"].astype(bool)] if len(predictions) else predictions
    one_track_fn = predictions[(predictions["true_label_id"] == 3) & (predictions["pred_label_id"] != 3)] if len(predictions) else predictions
    high_conf_wrong = wrong.sort_values("confidence", ascending=False).head(20) if len(wrong) else wrong
    low_conf_correct = predictions[predictions["correct"].astype(bool)].sort_values("confidence").head(20) if len(predictions) else predictions
    save_error_gallery(one_track_fn, errors_dir / "one_track_false_negative_gallery.png", "One Track False Negatives")
    save_error_gallery(high_conf_wrong, errors_dir / "high_confidence_wrong_gallery.png", "High-Confidence Wrong Predictions")
    save_error_gallery(low_conf_correct, errors_dir / "low_confidence_correct_gallery.png", "Low-Confidence Correct Predictions")
    if len(wrong):
        pairs = wrong.groupby(["true_label_name", "pred_label_name"]).size().reset_index(name="count")
        pairs["pair"] = pairs["true_label_name"] + " → " + pairs["pred_label_name"]
        plot_bar(pairs.sort_values("count", ascending=False).head(12), "pair", "count", errors_dir / "top_confused_class_pairs_barplot.png", "Top Confused Class Pairs", "Count")
        error_rate = predictions.groupby("true_label_name")["correct"].apply(lambda values: 1.0 - values.astype(bool).mean()).reset_index(name="error_rate")
        plot_bar(error_rate, "true_label_name", "error_rate", errors_dir / "classwise_error_rate_barplot.png", "Classwise Error Rate", "Error Rate")
    else:
        save_placeholder_figure(errors_dir / "top_confused_class_pairs_barplot.png", "Top Confused Class Pairs", "No errors available")
        save_placeholder_figure(errors_dir / "classwise_error_rate_barplot.png", "Classwise Error Rate", "No predictions available")
    save_placeholder_figure(errors_dir / "sankey_or_flow_confusion_plot.png", "Confusion Flow", "Sankey/flow plot is optional; confusion matrices contain the same counts.")


def save_model_comparison_placeholders(output_root: str | Path) -> None:
    root = ensure_dir(Path(output_root) / "plots" / "model_comparison")
    message = "Model comparison figures are generated after multiple model outputs are aggregated."
    for name in [
        "convnext_vs_dino_accuracy.png",
        "convnext_vs_dino_macro_f1.png",
        "convnext_vs_dino_one_track_recall.png",
        "convnext_vs_dino_calibration.png",
        "experiment_comparison_barplot.png",
        "experiment_comparison_radar_chart.png",
        "final_results_heatmap.png",
        "performance_vs_training_time.png",
        "performance_vs_model_size.png",
        "best_experiment_summary_plot.png",
    ]:
        save_placeholder_figure(root / name, name.replace("_", " ").replace(".png", "").title(), message)
