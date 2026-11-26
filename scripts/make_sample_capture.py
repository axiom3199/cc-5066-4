"""生成一个示例受控采集目录（含 views.csv 几何清单与占位 PNG）。

用法：
    python scripts/make_sample_capture.py <输出目录>

真实使用时，把占位 PNG 替换成同角度实拍图，并保持 views.csv 中的几何列
（角度单位为度）不变即可被 `python -m reflectance_fit.cli fit-dir` 读取。
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from reflectance_fit.observation import DEFAULT_PROTOCOL_DEG, render_observations
from reflectance_fit.params import clip_physical
from reflectance_fit.texture import synthetic_view_image


def main(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    gt = clip_physical({
        "albedo_r": 0.4, "albedo_g": 0.5, "albedo_b": 0.35,
        "roughness": 0.45, "specular": 0.55, "metallic": 0.0,
    })
    observations = render_observations(gt, DEFAULT_PROTOCOL_DEG)
    manifest = os.path.join(out_dir, "views.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "name", "file", "theta_view", "phi_view",
            "theta_light", "phi_light",
        ])
        for i, (obs, angles) in enumerate(zip(observations, DEFAULT_PROTOCOL_DEG)):
            tv, pv, tl, pl = angles
            fname = f"view_{i:02d}.png"
            Image.fromarray(synthetic_view_image(obs.radiance, 64, 64)).save(
                os.path.join(out_dir, fname)
            )
            writer.writerow([obs.name, fname, tv, pv, tl, pl])
    print(f"已生成示例采集目录: {out_dir}（{len(observations)} 个视图 + views.csv）")
    print("拟合命令: python -m reflectance_fit.cli fit-dir", out_dir)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
