import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


DEFAULT_ROUGH_ATTR = ["IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT"]
DEFAULT_DETAIL_ATTR = ["IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD"]
DEFAULT_WEIGHT_LEVEL = [0, 10, 15, 20, 25, 30]


def normalize_voyage_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def list_or_default(value: Any, default: List[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def voyage_rule_dict(value: Any, voyages: List[str], default: List[Any], *, fill_missing: bool = True) -> Dict[str, List[Any]]:
    normalized_voyages = [normalize_voyage_id(voyage) for voyage in voyages if normalize_voyage_id(voyage)]
    if isinstance(value, dict):
        out = {
            normalize_voyage_id(voyage): list_or_default(rules, default)
            for voyage, rules in value.items()
            if normalize_voyage_id(voyage)
        }
    elif value is None:
        out = {}
    else:
        shared = list_or_default(value, default)
        out = {voyage: list(shared) for voyage in normalized_voyages}
    if fill_missing:
        for voyage in normalized_voyages:
            out.setdefault(voyage, list(default))
    return out


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
        return {k: clean_for_json(v) for k, v in obj.items()}
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
    """Guandong input adapter used by the standalone flat yard planner."""

    def __init__(self):
        self.take_over_vessel: Dict[str, List] = {}       # 接管的航次信息
        self.bay_slots_detail: pd.DataFrame               # 空间场箱位信息
        self.tops_plan: pd.DataFrame = None               # tops上已有的计划占位信息
        self.area_function_info: pd.DataFrame = None      # 箱区功能和负载信息
        self.vessel_berth_info: pd.DataFrame = None       # 船舶靠泊信息
        self.planning_time: pd.Timestamp = pd.Timestamp.now()     # 做计划时间
        self.history_plan_info: pd.DataFrame = None       # 接管航次的上一次计划信息
        self.vessel_containers: Dict[str, Dict[str, pd.DataFrame|Dict]] = {}      # 航次的箱信息
        self.closed_area: Set[str] = set()                # 关闭的箱区
        self.berth_area_dist_matrix: pd.DataFrame = None  # 泊位箱区距离矩阵
        self.voyage_predict: Dict = {}                    # 航次预测信息
        self.large_plan: Dict = {}                        # 大计划计算结果
        self.adjust_plan_info: Dict = {}                  # 计划的人工调整信息
        self.user_design: Dict[str, bool] = {} # 用户指定大计划区域，区域放在user_design_large_plan_area,大中小计划都不能突破用户给定的区域
        self.user_design_large_plan_area: Dict[str, List[str]] = {} # 用户指定的大计划区域
        self.user_design_area_bay: Dict[str, Dict[str, List[str]]] = {} # 用户按航次、箱区限定中小计划可用贝位
        self.voyage_limit_areas: Dict[str, List[str]] = {} # 用户为航次限定的可选大范围箱区
        self.voyage_priority_areas: Dict[str, List[str]] = {} # 用户为航次限定的优选大范围箱区
        self.rough_attr: Dict[str, List[str]] = {}        # 粗属性分组依据
        self.detail_attr: Dict[str, List[str]] = {}       # 细分属性分组依据
        self.bay_rules: Dict[str, List[str]] = {}         # 贝不能混
        self.row_rules: Dict[str, List[str]] = {}         # 排不能混
        self.weight_level: Dict[str, List[int]] = {}      # 重量等级
        self.port_sail_area: Dict[str, List[str]] = {}    # 港口与航区的映射
        self.is_data_local: bool = False                  # 是否本地加载信息
        self.local_path: str = None                       # 本地加载文件地址
        self.need_save_data: bool = True                  # 是否保存输入信息

    def to_dict(self) -> dict:
        """Convert entire object to JSON-safe dictionary (recursive cleaning)."""
        return {
            "__class__": "InputAdapter",
            "take_over_vessel": self.take_over_vessel,
            "bay_slots_detail": clean_for_json(self.bay_slots_detail),
            "tops_plan": clean_for_json(self.tops_plan),
            "area_function_info": clean_for_json(self.area_function_info),
            "vessel_berth_info": clean_for_json(self.vessel_berth_info),
            "planning_time": self.planning_time,  # Handled by SafeJSONEncoder
            "history_plan_info": clean_for_json(self.history_plan_info),
            "vessel_containers": clean_for_json(self.vessel_containers),  # ✅ Recursive!
            "closed_area": list(self.closed_area),
            "berth_area_dist_matrix": clean_for_json(self.berth_area_dist_matrix),
            "voyage_predict": clean_for_json(self.voyage_predict),
            "large_plan": clean_for_json(self.large_plan),
            "adjust_plan_info": clean_for_json(self.adjust_plan_info),
            "user_design": self.user_design,
            "user_design_large_plan_area": self.user_design_large_plan_area,
            "user_design_area_bay": self.user_design_area_bay,
            "voyage_limit_areas": self.voyage_limit_areas,
            "voyage_priority_areas": self.voyage_priority_areas,
            "rough_attr": self.rough_attr,
            "detail_attr": self.detail_attr,
            "bay_rules": self.bay_rules,
            "row_rules": self.row_rules,
            "weight_level": self.weight_level,
            "port_sail_area": self.port_sail_area,
            "is_data_local": self.is_data_local,
            "local_path": self.local_path,
            "need_save_data": self.need_save_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InputAdapterGd":
        """Reconstruct object from dictionary (with DataFrame restoration)."""
        obj = cls()

        obj.take_over_vessel = data.get("take_over_vessel", {})
        obj.bay_slots_detail = restore_dataframe_from_split(data.get("bay_slots_detail"))
        obj.tops_plan = restore_dataframe_from_split(data.get("tops_plan"))
        obj.area_function_info = restore_dataframe_from_split(data.get("area_function_info"))
        obj.vessel_berth_info = restore_dataframe_from_split(data.get("vessel_berth_info"))

        pt = data.get("planning_time")
        obj.planning_time = pd.Timestamp(pt) if pt is not None else pd.NaT

        obj.history_plan_info = restore_dataframe_from_split(data.get("history_plan_info"))

        # 🔁 Restore nested DataFrames in vessel_containers
        vessel_containers_raw = data.get("vessel_containers", {})
        vessel_containers_restored = {}
        for vid, content in vessel_containers_raw.items():
            new_content = {}
            for key, value in content.items():
                if key in {"doc_cntrs"}:
                    new_content[key] = restore_dataframe_from_split(value)
                else:
                    new_content[key] = value  # keep as dict/list/scalar
            vessel_containers_restored[vid] = new_content
        obj.vessel_containers = vessel_containers_restored
        takeover_vessel_list = sum(obj.take_over_vessel.values(), [])
        obj.closed_area = set(data.get("closed_area", []))
        obj.berth_area_dist_matrix = restore_dataframe_from_split(data.get("berth_area_dist_matrix"))
        obj.voyage_predict = data.get("voyage_predict", {})
        obj.large_plan = data.get("large_plan", {})
        obj.adjust_plan_info = data.get("adjust_plan_info", {})
        obj.user_design = data.get("user_design", {})
        obj.user_design_large_plan_area = data.get("user_design_large_plan_area", {})
        obj.user_design_area_bay = data.get("user_design_area_bay", {})
        obj.voyage_limit_areas = data.get("voyage_limit_areas", {})
        obj.voyage_priority_areas = data.get("voyage_priority_areas", {})
        obj.rough_attr = voyage_rule_dict(data.get("rough_attr"), takeover_vessel_list, DEFAULT_ROUGH_ATTR)
        obj.detail_attr = voyage_rule_dict(data.get("detail_attr"), takeover_vessel_list, DEFAULT_DETAIL_ATTR)
        obj.bay_rules = voyage_rule_dict(data.get("bay_rules"), takeover_vessel_list, [], fill_missing=False)
        obj.row_rules = voyage_rule_dict(data.get("row_rules"), takeover_vessel_list, [], fill_missing=False)
        obj.weight_level = voyage_rule_dict(
            data.get("weight_level"),
            takeover_vessel_list,
            DEFAULT_WEIGHT_LEVEL,
            fill_missing=False,
        )
        obj.port_sail_area = data.get("port_sail_area", {})
        obj.is_data_local = data.get("is_data_local", False)
        obj.local_path = data.get("local_path")
        obj.need_save_data = data.get("need_save_data", True)

        return obj

    def save_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, cls=SafeJSONEncoder, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> "InputAdapterGd":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    json_path = ""
    data = InputAdapterGd.load_from_json(json_path)
