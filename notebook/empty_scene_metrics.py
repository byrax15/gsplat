import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from tqdm import tqdm


def alpha_blend(rgba, color):
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:]
    return rgb * alpha + color * (1 - alpha)


def get_gts():
    for scene in (
        pathlib.Path.home() / "NewVolume/SpacetimeGaussians/vdb_density_grad_prior"
    ).glob("*nobg*"):
        for frame in (scene / f"point").glob("colmap_*"):
            print("reading", frame)
            for camera in (frame / "images").iterdir():
                for bg in [[1e-6] * 3, [1.0] * 3]:
                    # im = plt.imread(camera)
                    yield {
                        "scene": scene.stem,
                        "object": "object" not in scene.stem,
                        "frame": frame.stem,
                        "camera": camera.stem,
                        "bg": bg,
                        "image": camera,
                    }


device = torch.device("cuda:0")
dtype = torch.float32

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device, dtype)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device, dtype)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device, dtype)


experiments = list(tqdm(get_gts(), desc="Collecting experiments"))
for i, d in tqdm(enumerate(experiments), "Computing metrics", total=len(experiments)):
    scene, object, frame, camera, bg, image = d.values()

    bg_color = torch.tensor(bg, device=device, dtype=dtype)
    gt = torch.tensor(plt.imread(image), device=device, dtype=dtype)
    gt_blend = alpha_blend(gt, bg_color)
    empty_im = torch.zeros_like(gt_blend)
    empty_im[..., :] = bg_color

    gt_blend = gt_blend.permute(2, 0, 1).unsqueeze(0)
    empty_im = empty_im.permute(2, 0, 1).unsqueeze(0)

    experiments[i]["psnr"] = psnr(empty_im, gt_blend).item()
    experiments[i]["ssim"] = ssim(empty_im, gt_blend).item()
    experiments[i]["lpips"] = lpips(empty_im, gt_blend).item()

metrics = pd.DataFrame.from_records(experiments)
metrics["bg"] = metrics["bg"].apply(lambda bg: str(bg))
metrics = metrics.pivot_table(
    index=["scene", "object", "frame", "camera", "bg"],
    values=["psnr", "ssim", "lpips"],
    aggfunc="mean",
).reset_index()
metrics.to_csv("empty_scene_metrics.csv", index=False)
