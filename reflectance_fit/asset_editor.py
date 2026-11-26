"""材质资产编辑器门面。

把旧工程装载、未确定粗糙度范围、灯光预览、下一观测选点和迁移接受证据
集中到一个有状态对象；GUI 与脚本都通过同一套真实评价路径调用。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evidence import MigrationDecision, accept_migration
from .lighting import (
    render_lighting_preview,
    select_next_observation,
)
from .project import MaterialProject, load_project


class MaterialAssetEditor:
    def __init__(self, project: Optional[str | MaterialProject] = None) -> None:
        self.project: Optional[MaterialProject] = None
        if project is not None:
            self.load_project(project)

    def load_project(self, project: str | MaterialProject) -> MaterialProject:
        if isinstance(project, MaterialProject):
            self.project = project
        else:
            self.project = load_project(project)
        return self.project

    # 语义别名，兼容不同界面命名。
    load = load_project

    @property
    def parameters(self) -> dict[str, float]:
        if self.project is None:
            from .params import default_parameters
            return default_parameters()
        return dict(self.project.parameters)

    @property
    def roughness_range(self) -> Optional[Tuple[float, float]]:
        if self.project is None:
            return None
        if self.project.roughness_range is not None:
            return self.project.roughness_range
        rough = self.project.parameters.get("roughness")
        return None if rough is None else (float(rough), float(rough))

    def lighting_preview(
        self,
        roughness_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        rng = roughness_range or self.roughness_range
        return render_lighting_preview(
            self.parameters, roughness_range=rng, **kwargs
        )

    def suggest_next_observation(
        self,
        errors: Any = None,
        *,
        observations=None,
        params: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ):
        return select_next_observation(
            errors,
            observations=observations,
            params=params or self.parameters,
            roughness_range=self.roughness_range,
            **kwargs,
        )

    def evaluate_migration(
        self,
        params: Mapping[str, float],
        *,
        train_observations=None,
        holdout_observations=None,
        measured_highlight_image: Optional[np.ndarray] = None,
        rendered_highlight_image: Optional[np.ndarray] = None,
        optimizer_evidence: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> MigrationDecision:
        return accept_migration(
            params,
            train_observations=train_observations,
            holdout_observations=holdout_observations,
            measured_highlight_image=measured_highlight_image,
            rendered_highlight_image=rendered_highlight_image,
            optimizer_evidence=optimizer_evidence,
            **kwargs,
        )


# 常见别名。
AssetEditor = MaterialAssetEditor
MaterialEditor = MaterialAssetEditor
