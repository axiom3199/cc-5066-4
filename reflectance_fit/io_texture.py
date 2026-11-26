"""从图像目录载入受控采集视图。

目录布局（几何必须显式提供，禁止猜测）::

    sample_dir/
      views.csv          # name,file,theta_view,phi_view,theta_light,phi_light
      view_00.png
      view_01.png
      ...

角度单位为度。缺少 ``views.csv`` 时抛出异常，由 GUI 状态栏提示——
本项目不会在几何未知的情况下拟合。
"""

from __future__ import annotations

import csv
import os
from typing import List

import numpy as np

from .texture import ViewImage


def load_view_directory(directory: str) -> List[ViewImage]:
    manifest = os.path.join(directory, "views.csv")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"目录 {directory} 缺少 views.csv 几何清单 "
            "(name,file,theta_view,phi_view,theta_light,phi_light)"
        )
    views: List[ViewImage] = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "file", "theta_view", "phi_view",
                    "theta_light", "phi_light"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"views.csv 列必须包含 {sorted(required)}")
        for row in reader:
            path = os.path.join(directory, row["file"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"清单引用的图像不存在: {path}")
            from PIL import Image
            img = np.asarray(Image.open(path).convert("RGB"))
            views.append(ViewImage(
                image_srgb=img,
                theta_view_deg=float(row["theta_view"]),
                phi_view_deg=float(row["phi_view"]),
                theta_light_deg=float(row["theta_light"]),
                phi_light_deg=float(row["phi_light"]),
                name=row["name"].strip() or os.path.basename(path),
            ))
    if len(views) < 4:
        raise ValueError(
            f"仅载入 {len(views)} 个视图，受控拟合至少需要 4 个不同角度。"
        )
    return views
