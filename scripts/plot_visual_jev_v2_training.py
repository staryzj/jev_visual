"""Render a GitHub- and manuscript-ready Visual-JEV V2 scaling figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment
from matplotlib.patches import Patch

SMALL = "#9CA3AF"
LARGE = "#2563EB"
ORANGE = "#D97706"
INK = "#172033"
MUTED = "#667085"
GRID = "#E7EAF0"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(report: dict[str, Any], section: str, key: str = "overall") -> float:
    value = report[section]
    if key != "overall":
        value = value[key]
    elif "overall" in value:
        value = value["overall"]
    return 100.0 * float(value["accuracy"])


def export_source_data(
    path: Path, small: dict[str, Any], large: dict[str, Any]
) -> None:
    rows: list[dict[str, Any]] = []
    for label, report in (("128", small), ("2048", large)):
        for item in report["history"]:
            for metric in (
                "train_loss",
                "candidate_loss",
                "visual_ranking_loss",
                "validation_accuracy",
                "validation_margin",
                "validation_loss",
            ):
                rows.append(
                    {
                        "section": "history",
                        "training_examples": label,
                        "epoch": item["epoch"],
                        "metric": metric,
                        "condition": "",
                        "category": "",
                        "value": item[metric],
                    }
                )
        for condition, metrics in report["after_training"].items():
            rows.append(
                {
                    "section": "visual_control",
                    "training_examples": label,
                    "epoch": "",
                    "metric": "accuracy",
                    "condition": condition,
                    "category": "",
                    "value": metrics["accuracy"],
                }
            )
        for category, metrics in report["sugarcrepe_after_training"].items():
            rows.append(
                {
                    "section": "sugarcrepe",
                    "training_examples": label,
                    "epoch": "",
                    "metric": "accuracy",
                    "condition": "after_training",
                    "category": category,
                    "value": metrics["accuracy"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#C7CDD8")
    axis.spines["bottom"].set_color("#C7CDD8")
    axis.tick_params(colors=MUTED, labelsize=7.2, length=3)
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.10,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def line(
    axis: plt.Axes,
    x: list[float],
    y: list[float],
    *,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    axis.plot(
        x,
        y,
        color=color,
        label=label,
        linewidth=2.0,
        marker=marker,
        markersize=3.8,
        markeredgewidth=0,
        linestyle=linestyle,
        zorder=3,
    )


def add_value_labels(axis: plt.Axes, bars: Any) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=INK,
        )


def make_figure(
    small: dict[str, Any],
    large: dict[str, Any],
    *,
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 7.6),
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.865,
        wspace=0.38,
        hspace=0.44,
    )
    for axis in axes.flat:
        style_axis(axis)

    histories = {"128 Examples": small["history"], "2,048 Examples": large["history"]}
    colors = {"128 Examples": SMALL, "2,048 Examples": LARGE}

    axis = axes[0, 0]
    for label, history in histories.items():
        line(
            axis,
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            color=colors[label],
            label=label,
        )
    axis.set(title="Objective convergence", xlabel="Epoch", ylabel="Total loss")
    panel_label(axis, "a")

    axis = axes[0, 1]
    large_history = large["history"]
    epochs = [row["epoch"] for row in large_history]
    line(
        axis,
        epochs,
        [row["candidate_loss"] for row in large_history],
        color=LARGE,
        label="Candidate CE",
    )
    line(
        axis,
        epochs,
        [row["visual_ranking_loss"] for row in large_history],
        color=ORANGE,
        label="Wrong-image ranking",
        marker="s",
    )
    axis.set(title="Expanded-run loss terms", xlabel="Epoch", ylabel="Loss")
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "b")

    axis = axes[0, 2]
    for label, history in histories.items():
        line(
            axis,
            [row["epoch"] for row in history],
            [100.0 * row["validation_accuracy"] for row in history],
            color=colors[label],
            label=label,
        )
    axis.axhline(100 / 3, color="#B9C0CC", linewidth=1, linestyle="--")
    axis.text(
        0.02,
        100 / 3 + 1.8,
        "3-way chance",
        transform=axis.get_yaxis_transform(),
        ha="left",
        color=MUTED,
        fontsize=6.8,
    )
    axis.set(
        title="Internal validation",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        ylim=(25, 102),
    )
    panel_label(axis, "c")

    axis = axes[1, 0]
    metrics = ["Internal", "SugarCrepe"]
    small_values = [
        accuracy(small, "after_training", "original"),
        accuracy(small, "sugarcrepe_after_training"),
    ]
    large_values = [
        accuracy(large, "after_training", "original"),
        accuracy(large, "sugarcrepe_after_training"),
    ]
    x = np.arange(len(metrics))
    width = 0.32
    small_bars = axis.bar(
        x - width / 2,
        small_values,
        width,
        color=SMALL,
        label="128 Examples",
        zorder=3,
    )
    large_bars = axis.bar(
        x + width / 2,
        large_values,
        width,
        color=LARGE,
        label="2,048 Examples",
        zorder=3,
    )
    add_value_labels(axis, small_bars)
    add_value_labels(axis, large_bars)
    axis.set(
        title="Final generalization",
        ylabel="Accuracy (%)",
        xticks=x,
        xticklabels=metrics,
        ylim=(0, 108),
    )
    panel_label(axis, "d")

    axis = axes[1, 1]
    categories = [
        key
        for key in large["sugarcrepe_after_training"]
        if key != "overall"
    ]
    labels = [key.replace("_", " ") for key in categories]
    y = np.arange(len(categories))
    small_category = [
        accuracy(small, "sugarcrepe_after_training", key) for key in categories
    ]
    large_category = [
        accuracy(large, "sugarcrepe_after_training", key) for key in categories
    ]
    for position, start, stop in zip(y, small_category, large_category):
        axis.plot([start, stop], [position, position], color="#D1D5DB", linewidth=1.4)
    axis.scatter(small_category, y, color=SMALL, s=25, label="128 Examples", zorder=3)
    axis.scatter(large_category, y, color=LARGE, s=28, label="2,048 Examples", zorder=4)
    axis.axvline(50, color="#B9C0CC", linewidth=1, linestyle="--")
    axis.set(
        title="SugarCrepe by category",
        xlabel="Accuracy (%)",
        yticks=y,
        yticklabels=labels,
        xlim=(20, 102),
    )
    axis.invert_yaxis()
    panel_label(axis, "e")

    axis = axes[1, 2]
    controls = ["original", "blank", "noise", "image_swap"]
    control_labels = ["Original", "Blank", "Noise", "Image swap"]
    small_controls = [
        100.0 * small["after_training"][key]["accuracy"] for key in controls
    ]
    large_controls = [
        100.0 * large["after_training"][key]["accuracy"] for key in controls
    ]
    x = np.arange(len(controls))
    small_bars = axis.bar(
        x - width / 2,
        small_controls,
        width,
        color=SMALL,
        label="128 Examples",
        zorder=3,
    )
    large_bars = axis.bar(
        x + width / 2,
        large_controls,
        width,
        color=LARGE,
        label="2,048 Examples",
        zorder=3,
    )
    axis.set(
        title="Visual-dependence controls",
        ylabel="Accuracy (%)",
        xticks=x,
        xticklabels=control_labels,
        ylim=(0, 105),
    )
    axis.tick_params(axis="x", labelrotation=18)
    axis.text(
        0.02,
        0.96,
        "Expanded run: 0.0 pp drop",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    panel_label(axis, "f")

    fig.suptitle(
        "Visual-JEV V2 | Scaling the candidate-aware adapter",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.936,
        "Frozen Qwen3-VL backbone · post-merger tokens · single deterministic run per scale",
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(facecolor=SMALL, edgecolor="none", label="128 Examples"),
            Patch(facecolor=LARGE, edgecolor="none", label="2,048 Examples"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.953),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.text(
        0.99,
        0.004,
        "No smoothing or error bars; values are preliminary single-seed results.",
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.flat),
        panel_ids=list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
        require_panel_labels=True,
        strict=True,
        json_out=output_stem.with_name(output_stem.name + "-alignment.json"),
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-report",
        type=Path,
        default=Path("reports/visual-jev-v2-coco-visual.json"),
    )
    parser.add_argument(
        "--large-report",
        type=Path,
        default=Path("reports/visual-jev-v2-coco-2k.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/visual-jev-v2-scaling"),
    )
    args = parser.parse_args()
    small = load_report(args.small_report)
    large = load_report(args.large_report)
    export_source_data(
        args.output.with_name(args.output.name + "-source-data.csv"),
        small,
        large,
    )
    make_figure(small, large, output_stem=args.output)


if __name__ == "__main__":
    main()
