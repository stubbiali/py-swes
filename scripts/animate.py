# -*- coding: utf-8 -*-
from __future__ import annotations
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import os
import subprocess
from typing import TYPE_CHECKING

import plot

if TYPE_CHECKING:
    from typing import Literal


def get_dataset_names(folder: str) -> list[str]:
    folder_abs = os.path.abspath(folder)
    all_names = (
        subprocess.run(["ls", folder_abs], capture_output=True).stdout.decode("utf-8").splitlines()
    )
    ds_names = [os.path.join(folder, name) for name in all_names if name.endswith(".nc")]
    return ds_names


def main(
    field_name: Literal["h", "u", "v"],
    folder: str,
    movie_name: str,
    *,
    color_map: str = "Spectral",
    figsize: tuple[int, int] = (10, 6),
    fps: int = 5,
) -> None:
    dataset_names = get_dataset_names(folder)

    writer = FFMpegWriter(fps=fps)

    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

    with writer.saving(fig, movie_name, len(dataset_names)):
        for i, dataset_name in enumerate(dataset_names):
            plot.main(
                field_name,
                dataset_name,
                dataset_ref_name=dataset_names[0],
                ax=ax,
                color_map=color_map,
                draw_colorbar=i == 0,
            )
            writer.grab_frame()


if __name__ == "__main__":
    main(
        field_name="h",
        folder="../output/williamson-1",
        movie_name="williamson_1.mp4",
        color_map="Reds",
        fps=15,
    )
