import json
import pathlib
import torch
import pandas as pd
import re


def get_stats():
    scene_match = re.compile(
        r"(?P<scene>(cloud[+]small)|(fire[+]smoke)|(teapot)|(cube))"
    )
    frame_no_match = re.compile(r"colmap_(?P<frame_no>\d+)")
    extra_indep_vars_matches = {
        k: (re.compile(v), default)
        for k, (v, default) in {
            "min-target": (r"min-target=(?P<mintarget>\d+)", None),
            "max-target": (r"max-target=(?P<maxtarget>\d+)", None),
            "prune-opa": (r"prune-opa=(?P<pruneopa>\d+)", 5e-3),
        }.items()
    }

    model = pathlib.Path("/home/aq85800/NewVolume/gsplat")
    for i, frame in enumerate(
        sorted(
            model.glob("budget+min-target=*+dual-bw-bg/*/colmap_*"),
            key=lambda f: f.name,
        )
    ):
        try:
            last_stats = sorted(
                frame.glob("*/stats/val_percam_step*.pt"),
                key=lambda f: f.stat().st_mtime,
            )[-1]
            averages = sorted(
                last_stats.parent.glob("val_step*.json"),
                key=lambda f: f.stat().st_mtime,
            )[-1]
        except IndexError:
            continue

        if match := scene_match.search(str(last_stats)):
            scene = match["scene"]
        else:
            scene = str(last_stats)

        if match := frame_no_match.search(str(last_stats)):
            frame_no = int(match["frame_no"])

        extra_indep_vars = {}
        for k, (matcher, default) in extra_indep_vars_matches.items():
            if match := matcher.search(str(last_stats)):
                extra_indep_vars[k] = int(match.group(1))
            elif default is not None:
                extra_indep_vars[k] = default

        stats = torch.load(last_stats, weights_only=False)
        psnrs = torch.stack(stats["psnr"])
        psnr_no_inf = psnrs[~torch.isinf(psnrs) & ~torch.isnan(psnrs)].mean().item()

        averages = json.load(averages.open())

        yield {
            "scene": scene,
            "frame": frame_no,
            **extra_indep_vars,
            **averages,
            "psnr_no_inf": psnr_no_inf,
        }


data = pd.DataFrame(get_stats())
print(data.to_csv(index=False))
