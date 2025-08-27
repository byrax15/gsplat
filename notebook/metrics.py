from glob import glob
import json
import pathlib
import torch
import numpy as np
import pandas as pd
import re
from tensorboard.backend.event_processing import event_accumulator


def get_stats(experiment_glob, float_vars, int_vars, str_vars, flag_vars):
    scene_match = re.compile(
        r"(?P<scene>(cloud[+]small)|(fire[+]smoke)|(teapot)|(cube))"
    )
    frame_no_match = re.compile(r"colmap_(?P<frame_no>\d+)")
    extra_indep_vars_matches = {
        k: (re.compile(v), default, success)
        for k, (v, default, success) in {
            **{
                f: (rf"{f}[=]?([+-]?\d*\.?\d+([eE][+-]?\d+)?)", None, None)
                for f in float_vars
            },
            **{i: (rf"{i}[=]?([+-]?\d+)", None, None) for i in int_vars},
            **{s: (rf"{s}[=]?(\w+)", None, None) for s in str_vars},
            **{f: (rf"({f})", False, True) for f in flag_vars},
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
            ea = event_accumulator.EventAccumulator(str(list(frame.glob("tb/*"))[-1]))
            _absorb_print = ea.Reload()
        except IndexError:
            continue

        if match := scene_match.search(str(last_stats)):
            scene = match["scene"]
        else:
            scene = str(last_stats)

        if match := frame_no_match.search(str(last_stats)):
            frame_no = int(match["frame_no"])

        extra_indep_vars = {}
        for k, (matcher, default, success) in extra_indep_vars_matches.items():
            if match := matcher.search(str(last_stats)):
                extra_indep_vars[k] = success or match.group(1)
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
            "time": frame.stem,
            "scene": scene,
            "frame": frame_no,
            **extra_indep_vars,
            **averages,
            "psnr_no_inf": psnr_no_inf,
            "loss": np.mean([s.value for s in ea.Scalars("train/loss")]),
        }


def main(
    experiment_glob: str,
    /,
    float_vars: list[str] = [],
    int_vars: list[str] = [],
    str_vars: list[str] = [],
    flag_vars: list[str] = [],
):
    data = pd.DataFrame(
        get_stats(experiment_glob, float_vars, int_vars, str_vars, flag_vars)
    )
    print(data.to_csv(index=False, decimal=",", sep=";"))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
