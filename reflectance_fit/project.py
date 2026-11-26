"""材质工程文件的版本化与旧工程兼容装载。

新版本文件记录优化后端契约和真实重渲染证据；旧工程可能没有这些字段，
装载时只做归一化并给出迁移警告，绝不要求手工改数据。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .optimization import OPTIMIZER_CONTRACT_VERSION

PROJECT_SCHEMA_VERSION = "2.0.0"
PROJECT_FORMAT = "controlled-reflectance-material"
LEGACY_VERSIONS = {"", "0", "0.1", "0.1.0", "1", "1.0", "1.0.0"}


@dataclass
class MaterialProject:
    path: Optional[str]
    schema: str
    version: str
    original_version: str
    migrated: bool
    parameters: Dict[str, float]
    fixed: Dict[str, float] = field(default_factory=dict)
    active_params: Tuple[str, ...] = ()
    roughness_range: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    optimizer_contract: str = OPTIMIZER_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": PROJECT_SCHEMA_VERSION,
            "optimizer_contract": self.optimizer_contract,
            "parameters": self.parameters,
            "fixed": self.fixed,
            "active_params": list(self.active_params),
            "roughness_range": None if self.roughness_range is None
            else list(self.roughness_range),
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if target is None:
            raise ValueError("尚未指定保存路径")
        save_project(self, target)
        self.path = os.path.abspath(target)
        return target


def _as_version(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value).strip()


def _find_parameter_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("parameters", "params", "fitted_parameters", "material",
                "values", "pbr", "shader"):
        value = data.get(key)
        if isinstance(value, Mapping):
            # 有些旧文件把 PBR 参数再嵌一层。
            for nested in ("parameters", "params", "pbr", "values"):
                inner = value.get(nested)
                if isinstance(inner, Mapping) and any(
                    k in inner for k in ("roughness", "specular", "albedo_r")
                ):
                    return inner
            if any(k in value for k in
                   ("roughness", "specular", "albedo_r", "albedo", "metallic")):
                return value
    # 扁平旧格式：参数直接位于顶层。
    if any(k in data for k in ("roughness", "specular", "albedo_r")):
        return data
    return {}


def _normalize_parameters(raw_params: Mapping[str, Any],
                          warnings: list[str],
                          *,
                          fill_defaults: bool = False) -> Dict[str, float]:
    from .params import PARAM_SPEC, clip_physical, default_parameters

    flat: Dict[str, float] = dict(default_parameters()) if fill_defaults else {}
    if "albedo" in raw_params and isinstance(raw_params["albedo"], (list, tuple)):
        rgb = raw_params["albedo"]
        flat.update(albedo_r=float(rgb[0]), albedo_g=float(rgb[1]),
                    albedo_b=float(rgb[2]))
    for key in PARAM_SPEC:
        if key in raw_params:
            flat[key] = float(raw_params[key])
    # 兼容 albedo=[r,g,b] 之外的颜色字段。
    for src, dst in (("albedoR", "albedo_r"), ("albedoG", "albedo_g"),
                     ("albedoB", "albedo_b")):
        if src in raw_params:
            flat[dst] = float(raw_params[src])

    before = dict(flat)
    if fill_defaults:
        normalized = clip_physical(flat)
    else:
        normalized = {
            key: PARAM_SPEC[key].clip(float(value))
            for key, value in flat.items()
            if key in PARAM_SPEC
        }
    for key, value in before.items():
        pr = PARAM_SPEC[key]
        if value < pr.low or value > pr.high:
            warnings.append(
                f"旧工程参数 {key}={value} 超出当前范围 "
                f"[{pr.low},{pr.high}]，装载时已裁剪并保留在 raw 中"
            )
    return normalized


def _find_roughness_range(data: Mapping[str, Any],
                          params: Mapping[str, float]) -> Optional[Tuple[float, float]]:
    for key in ("roughness_range", "roughnessRange", "roughness_bounds",
                "undetermined_roughness_range", "feasible_roughness"):
        value = data.get(key)
        if isinstance(value, Mapping):
            lo = value.get("low", value.get("min", value.get("lo")))
            hi = value.get("high", value.get("max", value.get("hi")))
            if lo is not None and hi is not None:
                return (float(lo), float(hi))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    feasible = data.get("feasible_ranges")
    if isinstance(feasible, Mapping) and "roughness" in feasible:
        value = feasible["roughness"]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    if data.get("roughness_undetermined") is True:
        from .params import ROUGHNESS_RANGE
        return (ROUGHNESS_RANGE.low, ROUGHNESS_RANGE.high)
    if "roughness_min" in data and "roughness_max" in data:
        return (float(data["roughness_min"]), float(data["roughness_max"]))
    return None


def load_project(path: str) -> MaterialProject:
    """装载新旧材质工程；文件不存在或 JSON 损坏时抛出明确异常。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"材质工程不存在: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, Mapping):
        raise ValueError("材质工程顶层必须是 JSON 对象")

    warnings: list[str] = []
    original_version = _as_version(
        data.get("version", data.get("format_version", data.get("schema_version")))
    )
    version = original_version or "legacy-unversioned"
    migrated = original_version in LEGACY_VERSIONS or not original_version.startswith("2")
    if migrated:
        warnings.append(
            f"旧材质工程版本 {version} 已兼容装载；保存时将迁移到 "
            f"{PROJECT_SCHEMA_VERSION}，优化契约 {OPTIMIZER_CONTRACT_VERSION}"
        )

    raw_params = _find_parameter_mapping(data)
    parameters = _normalize_parameters(raw_params, warnings, fill_defaults=True)
    fixed_raw = data.get("fixed", data.get("fixed_parameters", {}))
    fixed: Dict[str, float] = {}
    if isinstance(fixed_raw, Mapping):
        fixed = _normalize_parameters(fixed_raw, [])
    active_raw = data.get("active_params")
    if isinstance(active_raw, (list, tuple)):
        active_params = tuple(str(k) for k in active_raw)
    else:
        from .params import PARAM_SPEC
        active_params = tuple(k for k in PARAM_SPEC if k not in fixed)
    roughness_range = _find_roughness_range(data, parameters)

    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    return MaterialProject(
        path=os.path.abspath(path),
        schema=str(data.get("schema", data.get("format", PROJECT_FORMAT))),
        version=PROJECT_SCHEMA_VERSION if migrated else version,
        original_version=version,
        migrated=migrated,
        parameters=parameters,
        fixed=fixed,
        active_params=active_params,
        roughness_range=roughness_range,
        metadata=dict(metadata),
        raw=dict(data),
        warnings=warnings,
    )


load_material_project = load_project
load_material_asset = load_project


def save_project(project: MaterialProject | Mapping[str, Any], path: str) -> str:
    if isinstance(project, MaterialProject):
        payload = project.to_dict()
    else:
        payload = dict(project)
        payload.setdefault("schema", PROJECT_FORMAT)
        payload["version"] = PROJECT_SCHEMA_VERSION
        payload["optimizer_contract"] = OPTIMIZER_CONTRACT_VERSION
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


save_material_project = save_project
