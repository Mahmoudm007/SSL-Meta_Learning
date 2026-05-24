from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.class_mapping import ClassMapping
from src.utils.io import ensure_dir, write_dataframe


def extract_embeddings(model: torch.nn.Module, dataset, device: torch.device, batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    embeddings = []
    labels = []
    paths: list[str] = []
    with torch.no_grad():
        for images, batch_labels, batch_paths, _ in tqdm(loader, desc="Extract embeddings", leave=False):
            images = images.to(device)
            features = model.forward_features(images)
            embeddings.append(features.detach().cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            paths.extend(list(batch_paths))
    if not embeddings:
        return np.empty((0, 0)), np.array([], dtype=int), []
    return np.concatenate(embeddings, axis=0), np.asarray(labels, dtype=int), paths


def save_embedding_outputs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: Sequence[str],
    class_mapping: ClassMapping,
    output_dir: str | Path,
    predictions: pd.DataFrame | None = None,
) -> None:
    root = ensure_dir(output_dir)
    if embeddings.size == 0:
        _placeholder(root / "umap_embeddings.png", "No embeddings available")
        _placeholder(root / "tsne_embeddings.png", "No embeddings available")
        _placeholder(root / "pca_embeddings.png", "No embeddings available")
        return
    columns = {f"emb_{index}": embeddings[:, index] for index in range(embeddings.shape[1])}
    frame = pd.DataFrame(columns)
    frame.insert(0, "image_path", list(paths))
    frame.insert(1, "label_id", labels)
    frame.insert(2, "label_name", [class_mapping.id_to_name.get(int(label), str(label)) for label in labels])
    write_dataframe(root / "embeddings.csv", frame)
    centroids = _centroids(embeddings, labels, class_mapping)
    write_dataframe(root / "class_centroids.csv", centroids)
    distance_matrix = _centroid_distance_matrix(centroids)
    write_dataframe(root / "centroid_distance_matrix.csv", distance_matrix)
    _heatmap(distance_matrix, root / "centroid_distance_heatmap.png", "Class Centroid Distance")
    _heatmap(distance_matrix, root / "class_centroid_distance_heatmap.png", "Class Centroid Distance")
    _intra_inter_plot(embeddings, labels, root / "intra_inter_class_distance_plot.png")
    _projection_plot(embeddings, labels, class_mapping, root / "pca_embeddings.png", "PCA Embeddings", "pca")
    _projection_plot(embeddings, labels, class_mapping, root / "tsne_embeddings.png", "t-SNE Embeddings", "tsne")
    _projection_plot(embeddings, labels, class_mapping, root / "umap_embeddings.png", "UMAP Embeddings", "umap")
    _projection_plot(embeddings, labels, class_mapping, root / "embedding_by_correctness_umap.png", "Embedding by Correctness", "umap", predictions=predictions, color_by="correct")
    _projection_plot(embeddings, labels, class_mapping, root / "embedding_by_confidence_umap.png", "Embedding by Confidence", "umap", predictions=predictions, color_by="confidence")
    _projection_plot(embeddings, labels, class_mapping, root / "one_track_neighborhood_umap.png", "One Track Neighborhood", "umap", highlight_one_track=True)
    _projection_plot(embeddings, labels, class_mapping, root / "hard_negative_pair_embedding_plot.png", "Hard Negative Pair Embedding", "pca")
    _placeholder(root / "nearest_neighbor_retrieval_examples.png", "Nearest-neighbor examples are listed in embeddings.csv; image gallery generation is dataset-size dependent.")


def _centroids(embeddings: np.ndarray, labels: np.ndarray, class_mapping: ClassMapping) -> pd.DataFrame:
    rows = []
    for class_id, class_name in enumerate(class_mapping.class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue
        centroid = embeddings[mask].mean(axis=0)
        row = {"label_id": class_id, "label_name": class_name}
        for index, value in enumerate(centroid):
            row[f"emb_{index}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _centroid_distance_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    if centroids.empty:
        return pd.DataFrame()
    vectors = centroids[[column for column in centroids.columns if column.startswith("emb_")]].to_numpy()
    distances = cdist(vectors, vectors, metric="euclidean")
    return pd.DataFrame(distances, index=centroids["label_name"], columns=centroids["label_name"]).reset_index(names="label_name")


def _projection(embeddings: np.ndarray, method: str) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2))
    if method == "tsne" and embeddings.shape[0] >= 4:
        perplexity = min(30, max(2, embeddings.shape[0] // 4))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(embeddings)
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
        except Exception:
            return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def _projection_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_mapping: ClassMapping,
    path: Path,
    title: str,
    method: str,
    predictions: pd.DataFrame | None = None,
    color_by: str | None = None,
    highlight_one_track: bool = False,
) -> None:
    coords = _projection(embeddings, method)
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_by == "confidence" and predictions is not None and "confidence" in predictions:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=predictions["confidence"].astype(float), cmap="viridis", s=10)
        fig.colorbar(scatter, ax=ax, label="confidence")
    elif color_by == "correct" and predictions is not None and "correct" in predictions:
        colors = predictions["correct"].astype(bool).map({True: "#2ca02c", False: "#d62728"})
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10)
    elif highlight_one_track:
        colors = np.where(labels == 3, "#d62728", "#9aa0a6")
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10)
    else:
        for class_id, class_name in enumerate(class_mapping.class_names):
            mask = labels == class_id
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=10, label=class_name)
        ax.legend(fontsize=7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if frame.empty:
        ax.text(0.5, 0.5, "No centroid distances available", ha="center", va="center")
    else:
        matrix = frame.drop(columns=["label_name"]).to_numpy()
        image = ax.imshow(matrix, cmap="magma")
        fig.colorbar(image, ax=ax)
        labels = frame["label_name"].tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _intra_inter_plot(embeddings: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(labels) < 2:
        ax.text(0.5, 0.5, "Not enough embeddings", ha="center", va="center")
    else:
        distances = cdist(embeddings, embeddings)
        same = labels[:, None] == labels[None, :]
        intra = distances[same & ~np.eye(len(labels), dtype=bool)]
        inter = distances[~same]
        ax.boxplot([intra[:5000], inter[:5000]], labels=["intra", "inter"])
        ax.set_ylabel("Euclidean distance")
    ax.set_title("Intra-/Inter-Class Distances")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _placeholder(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
