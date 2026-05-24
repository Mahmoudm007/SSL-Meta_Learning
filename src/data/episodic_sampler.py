from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from src.data.class_mapping import HARD_CLASS_PAIRS, ONE_TRACK_CLASS_ID, RSC_CLASS_NAMES
from src.data.datasets import ImageRecord


@dataclass
class Episode:
    support_indices: List[int]
    query_indices: List[int]
    class_ids: List[int]
    hard: bool
    hard_pair: tuple[int, int] | None = None


class EpisodicSampler:
    def __init__(
        self,
        records: Sequence[ImageRecord],
        support_per_class: int,
        query_per_class: int,
        class_ids: Sequence[int] | None = None,
        hard_episode_probability: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.support_per_class = support_per_class
        self.query_per_class = query_per_class
        self.class_ids = list(class_ids) if class_ids is not None else sorted({record.label_id for record in records})
        self.hard_episode_probability = hard_episode_probability
        self.rng = random.Random(seed)
        self.indices_by_class: Dict[int, List[int]] = {class_id: [] for class_id in self.class_ids}
        for index, record in enumerate(self.records):
            if record.label_id in self.indices_by_class:
                self.indices_by_class[record.label_id].append(index)
        missing = [class_id for class_id, indices in self.indices_by_class.items() if not indices]
        if missing:
            raise ValueError(f"Cannot build episodes because these class ids have no records: {missing}")

    def _sample_indices(self, class_id: int, count: int) -> List[int]:
        candidates = self.indices_by_class[class_id]
        if len(candidates) >= count:
            return self.rng.sample(candidates, count)
        return [self.rng.choice(candidates) for _ in range(count)]

    def _hard_pair(self) -> tuple[int, int]:
        name_to_id = {name: index for index, name in enumerate(RSC_CLASS_NAMES)}
        valid_pairs = [
            (name_to_id[left], name_to_id[right])
            for left, right in HARD_CLASS_PAIRS
            if name_to_id[left] in self.class_ids and name_to_id[right] in self.class_ids
        ]
        if not valid_pairs:
            return (ONE_TRACK_CLASS_ID, ONE_TRACK_CLASS_ID)
        one_track_pairs = [pair for pair in valid_pairs if ONE_TRACK_CLASS_ID in pair]
        if one_track_pairs and self.rng.random() < 0.75:
            return self.rng.choice(one_track_pairs)
        return self.rng.choice(valid_pairs)

    def sample(self) -> Episode:
        hard = self.rng.random() < self.hard_episode_probability
        hard_pair = self._hard_pair() if hard else None
        class_ids = list(self.class_ids)
        support_indices: List[int] = []
        query_indices: List[int] = []
        for class_id in class_ids:
            sampled = self._sample_indices(class_id, self.support_per_class + self.query_per_class)
            support_indices.extend(sampled[: self.support_per_class])
            query_indices.extend(sampled[self.support_per_class :])
        self.rng.shuffle(support_indices)
        self.rng.shuffle(query_indices)
        return Episode(support_indices, query_indices, class_ids, hard=hard, hard_pair=hard_pair)

    def sample_many(self, count: int) -> Iterable[Episode]:
        for _ in range(count):
            yield self.sample()
