from dataclasses import dataclass
from glob import glob
import json
import pathlib
from typing import Any
import torch
import pandas as pd
import re
import sys


def get_stats(experiment_glob: str):
    scene_match = re.compile(
        r"(?P<scene>(cloud[+]small)|(fire[+]smoke)|(teapot)|(cube))"
    )
    frame_no_match = re.compile(r"colmap_(?P<frame_no>\d+)")
    extra_indep_vars_matches = {
        k: (re.compile(v), default)
        for k, (v, default) in {
            # "min-target": (r"min-target=(?P<mintarget>\d+)", None),
            # "max-target": (r"max-target=(?P<maxtarget>\d+)", None),
            # "prune-opa": (r"prune-opa=(?P<pruneopa>[+-]?\d*\.?\d+([eE][+-]?\d+)?)", Exception('Prune-Opa Not Found')),
            # "density-reg": (
            #     r"density-reg=(?P<densityreg>[+-]?\d*\.?\d+([eE][+-]?\d+)?)",
            #     None,
            # ),
            # "density-cells": (r"density-cells=(?P<densitycells>\d+)", None),
        }.items()
    }

    for i, frame in enumerate(
        sorted(
            [pathlib.Path(f) for f in glob(experiment_glob)],
            key=lambda f: f.name,
        )
    ):
        try:
            last_stats = sorted(
                frame.glob("stats/val_percam_step*.pt"),
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
                extra_indep_vars[k] = match.group(1)
            elif isinstance(default, Exception):
                default.add_note("\tin line: " + str(last_stats))
                raise default
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


def main(
    experiment_glob: str,
    /,
):
    data = pd.DataFrame(get_stats(experiment_glob))
    print(data.to_csv(index=False))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
