import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


def normalize_voyage_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        df_clean = obj.astype(object).where(pd.notna(obj), None)
        return df_clean.to_dict(orient="split")
    if isinstance(obj, pd.Series):
        if obj.empty:
            return None
        series_clean = obj.astype(object).where(pd.notna(obj), None)
        return series_clean.to_dict()
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    return obj


def restore_dataframe_from_split(data: Optional[Dict]) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    try:
        return pd.DataFrame(data=data.get("data", []), index=data.get("index"), columns=data.get("columns"))
    except Exception:
        return pd.DataFrame()


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return None if pd.isna(obj) else obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class InputAdapterGd:
    """JSON adapter containing only inputs used by the yard-planning model."""

    def __init__(self):
        self.take_over_vessel: Dict[str, List] = {}
        self.bay_slots_detail: Optional[pd.DataFrame] = None
        self.tops_plan: Optional[pd.DataFrame] = None
        self.area_function_info: Optional[pd.DataFrame] = None
        self.vessel_berth_info: Optional[pd.DataFrame] = None
        self.planning_time: pd.Timestamp = pd.Timestamp.now()
        self.vessel_containers: Dict[str, Dict[str, pd.DataFrame | Dict]] = {}
        self.closed_area: Set[str] = set()
        self.berth_area_dist_matrix: Optional[pd.DataFrame] = None
        self.large_plan: Dict = {}

    def to_dict(self) -> dict:
        """Convert the current model inputs to a JSON-safe dictionary."""
        return {
            "take_over_vessel": self.take_over_vessel,
            "bay_slots_detail": clean_for_json(self.bay_slots_detail),
            "tops_plan": clean_for_json(self.tops_plan),
            "area_function_info": clean_for_json(self.area_function_info),
            "vessel_berth_info": clean_for_json(self.vessel_berth_info),
            "planning_time": self.planning_time,
            "vessel_containers": clean_for_json(self.vessel_containers),
            "closed_area": list(self.closed_area),
            "berth_area_dist_matrix": clean_for_json(self.berth_area_dist_matrix),
            "large_plan": clean_for_json(self.large_plan),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InputAdapterGd":
        """Reconstruct the current model inputs from a dictionary."""
        obj = cls()
        obj.take_over_vessel = data.get("take_over_vessel", {})
        obj.bay_slots_detail = restore_dataframe_from_split(data.get("bay_slots_detail"))
        obj.tops_plan = restore_dataframe_from_split(data.get("tops_plan"))
        obj.area_function_info = restore_dataframe_from_split(data.get("area_function_info"))
        obj.vessel_berth_info = restore_dataframe_from_split(data.get("vessel_berth_info"))

        planning_time = data.get("planning_time")
        obj.planning_time = pd.Timestamp(planning_time) if planning_time is not None else pd.NaT

        vessel_containers = {}
        for voyage_id, content in data.get("vessel_containers", {}).items():
            restored = {}
            for key, value in content.items():
                restored[key] = restore_dataframe_from_split(value) if key == "doc_cntrs" else value
            vessel_containers[voyage_id] = restored
        obj.vessel_containers = vessel_containers

        obj.closed_area = set(data.get("closed_area", []))
        obj.berth_area_dist_matrix = restore_dataframe_from_split(data.get("berth_area_dist_matrix"))
        obj.large_plan = data.get("large_plan", {})
        return obj

    def save_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, cls=SafeJSONEncoder, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> "InputAdapterGd":
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)
