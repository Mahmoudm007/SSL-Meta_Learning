from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


RSC_CLASS_NAMES: List[str] = [
    "0 Bare",
    "1 Centre - Partly",
    "2 Two Track - Partly",
    "3 One Track - Partly",
    "4 Fully",
]

BLACK_ICE_CLASS_NAME = "5 Black Ice"

SHORT_CLASS_NAMES: Dict[str, str] = {
    "0 Bare": "Bare",
    "1 Centre - Partly": "Centre_Partly",
    "2 Two Track - Partly": "Two_Track_Partly",
    "3 One Track - Partly": "One_Track_Partly",
    "4 Fully": "Fully",
    "5 Black Ice": "Black_Ice",
}

ONE_TRACK_CLASS_ID = 3

KNOWN_CLASS_COUNTS: Dict[str, int] = {
    "0 Bare": 3335,
    "1 Centre - Partly": 1505,
    "2 Two Track - Partly": 2191,
    "3 One Track - Partly": 434,
    "4 Fully": 3139,
}

HARD_CLASS_PAIRS = [
    ("3 One Track - Partly", "1 Centre - Partly"),
    ("3 One Track - Partly", "2 Two Track - Partly"),
    ("3 One Track - Partly", "4 Fully"),
    ("1 Centre - Partly", "2 Two Track - Partly"),
    ("2 Two Track - Partly", "4 Fully"),
]


@dataclass(frozen=True)
class ClassMapping:
    class_names: List[str]

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {index: name for index, name in enumerate(self.class_names)}

    @property
    def short_names(self) -> List[str]:
        return [SHORT_CLASS_NAMES.get(name, name.replace(" ", "_").replace("-", "")) for name in self.class_names]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "class_names": self.class_names,
            "name_to_id": self.name_to_id,
            "id_to_name": {str(k): v for k, v in self.id_to_name.items()},
            "short_names": self.short_names,
        }


def default_class_mapping(include_black_ice: bool = False) -> ClassMapping:
    names = list(RSC_CLASS_NAMES)
    if include_black_ice:
        names.append(BLACK_ICE_CLASS_NAME)
    return ClassMapping(names)


def parse_class_name(value: str, class_names: Iterable[str] | None = None) -> str:
    names = list(class_names or RSC_CLASS_NAMES)
    normalized = value.strip().lower()
    for name in names:
        if normalized == name.lower() or normalized == name.split(" ", 1)[-1].lower():
            return name
    raise ValueError(f"Unknown class '{value}'. Expected one of: {names}")


def sanitize_class_name(name: str) -> str:
    return SHORT_CLASS_NAMES.get(name, name.replace(" ", "_").replace("-", "_"))
