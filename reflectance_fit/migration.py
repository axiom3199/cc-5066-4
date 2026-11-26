"""优化后端迁移对照与材质接受流程。

版本说明
--------
``OPTIMIZER_MIGRATION_VERSION`` 标识反射拟合已从迁移前的直接
logit/L-BFGS-B 调用，迁移到 :mod:`reflectance_fit.optimization` 的显式
边界/梯度/停止条件适配层。旧后端保留为 ``legacy``，用于旧样例和契约对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .evidence import MigrationDecision, accept_migration
from .fitter import FitResult, fit_brdf
from .geometry import Observation
from .optimization import DEFAULT_BACKEND_ID, LEGACY_BACKEND_ID

OPTIMIZER_MIGRATION_VERSION = "reflection-optimizer-backend-migration/v2.0"
MIGRATION_VERSION = OPTIMIZER_MIGRATION_VERSION


@dataclass
class BackendComparison:
    old: FitResult
    new: FitResult
    both_explain_training: bool
    training_rmse_relative_difference: float
    migration_version: str = OPTIMIZER_MIGRATION_VERSION
    decision: Optional[MigrationDecision] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_version": self.migration_version,
            "old_backend": self.old.backend,
            "new_backend": self.new.backend,
            "old_train_rmse": self.old.train_rmse,
            "new_train_rmse": self.new.train_rmse,
            "old_holdout_rmse": self.old.holdout_rmse,
            "new_holdout_rmse": self.new.holdout_rmse,
            "both_explain_training": self.both_explain_training,
            "training_rmse_relative_difference":
                self.training_rmse_relative_difference,
            "decision": None if self.decision is None else self.decision.to_dict(),
            "notes": self.notes,
        }


def compare_optimizer_backends(
    train_obs: Sequence[Observation],
    holdout_obs: Sequence[Observation],
    *,
    fixed: Optional[Mapping[str, float]] = None,
    training_equivalence_tolerance: float = 0.25,
    training_explanation_limit: float = 0.02,
    measured_highlight_image: Optional[np.ndarray] = None,
    rendered_highlight_image: Optional[np.ndarray] = None,
    highlight_size_tolerance: float = 0.35,
    maxiter: int = 200,
    **highlight_kwargs: Any,
) -> BackendComparison:
    """运行旧与新求解路径，并按留出光位实际高光决定是否接受新材质。"""
    old_result = fit_brdf(
        train_obs, holdout_obs, fixed=fixed, maxiter=maxiter,
        backend=LEGACY_BACKEND_ID,
    )
    new_result = fit_brdf(
        train_obs, holdout_obs, fixed=fixed, maxiter=maxiter,
        backend=DEFAULT_BACKEND_ID,
    )
    denom = max(min(old_result.train_rmse, new_result.train_rmse), 1e-12)
    rel_diff = abs(old_result.train_rmse - new_result.train_rmse) / denom
    both_explain = (
        old_result.train_rmse <= training_explanation_limit
        and new_result.train_rmse <= training_explanation_limit
        and rel_diff <= training_equivalence_tolerance
    )
    notes: list[str] = []
    if not both_explain:
        notes.append("新旧后端对训练样片的解释不等价；应先检查求解契约。")
    if measured_highlight_image is None or rendered_highlight_image is None:
        notes.append("缺少留出光位实测/重渲染高光图像；后端对照不能替代高光验收。")

    decision = accept_migration(
        new_result.parameters,
        train_observations=train_obs,
        holdout_observations=holdout_obs,
        measured_highlight_image=measured_highlight_image,
        rendered_highlight_image=rendered_highlight_image,
        highlight_size_tolerance=highlight_size_tolerance,
        optimizer_evidence=new_result.optimizer_evidence,
        active=new_result.active_params,
        **highlight_kwargs,
    )
    if both_explain and decision.highlight.get("size_match") is True:
        notes.append("训练解释等价；接受依据是留出光位实际高光尺寸与独立重渲染误差。")
    return BackendComparison(
        old=old_result,
        new=new_result,
        both_explain_training=both_explain,
        training_rmse_relative_difference=float(rel_diff),
        decision=decision,
        notes=notes,
    )


# 常见语义别名。
compare_backends = compare_optimizer_backends
migration_check = compare_optimizer_backends
verify_optimizer_migration = compare_optimizer_backends


def migration_version() -> str:
    return OPTIMIZER_MIGRATION_VERSION
