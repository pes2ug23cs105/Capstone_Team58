from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_FIGURES_DIR = Path("figures")


def _validate_sample(sample: dict) -> None:
    required = {"dataset", "steps", "causal_scores", "T_star", "is_correct"}
    missing = required.difference(sample.keys())
    if missing:
        raise KeyError(f"Sample is missing required keys: {sorted(missing)}")


def _group_samples_by_dataset(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        _validate_sample(sample)
        grouped[str(sample["dataset"])].append(sample)
    return dict(grouped)


def _max_step_from_samples(samples: list[dict]) -> int:
    max_step = 0
    for sample in samples:
        steps = sample.get("steps", [])
        if steps:
            max_step = max(max_step, max(int(x) for x in steps))
        t_star = int(sample.get("T_star", 0))
        max_step = max(max_step, t_star)
    return max_step


def _plot_causal_decay_curves(grouped: dict[str, list[dict]], figures_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    for dataset, ds_samples in sorted(grouped.items()):
        scores_by_step: dict[int, list[float]] = defaultdict(list)

        for sample in ds_samples:
            steps = sample["steps"]
            scores = sample["causal_scores"]
            if len(steps) != len(scores):
                continue

            for step, score in zip(steps, scores):
                scores_by_step[int(step)].append(float(score))

        if not scores_by_step:
            continue

        x_vals = sorted(scores_by_step.keys())
        y_vals = [sum(scores_by_step[x]) / len(scores_by_step[x]) for x in x_vals]
        plt.plot(x_vals, y_vals, marker="o", label=dataset)

    plt.xlabel("Step Number")
    plt.ylabel("Average Causal Necessity Score")
    plt.title("Causal Necessity Score Decay Curves")
    plt.legend()
    plt.grid(True, alpha=0.25)

    out_path = figures_dir / "causal_decay_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_t_star_histograms(grouped: dict[str, list[dict]], max_step: int, figures_dir: Path) -> None:
    datasets = sorted(grouped.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Center integer-valued T* in bins: [-0.5, 0.5), [0.5, 1.5), ...
    bins = [i - 0.5 for i in range(0, max_step + 2)]

    for i, ax in enumerate(axes):
        if i < len(datasets):
            dataset = datasets[i]
            t_stars = [int(s["T_star"]) for s in grouped[dataset]]
            ax.hist(t_stars, bins=bins)
            ax.set_title(dataset)
            ax.set_xlabel("T* (Step Index)")
            ax.set_xlim(-0.5, max_step + 0.5)
        else:
            ax.set_axis_off()

    axes[0].set_ylabel("Count of Samples")
    fig.suptitle("T* Distribution Histogram")

    out_path = figures_dir / "t_star_histograms.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _error_rate(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    incorrect = sum(1 for s in samples if not bool(s["is_correct"]))
    return incorrect / len(samples)


def _plot_t_star_error_correlation(grouped: dict[str, list[dict]], figures_dir: Path) -> None:
    datasets = sorted(grouped.keys())

    early_rates: list[float] = []
    late_rates: list[float] = []

    for dataset in datasets:
        ds_samples = grouped[dataset]
        if not ds_samples:
            early_rates.append(0.0)
            late_rates.append(0.0)
            continue

        ranked = sorted(ds_samples, key=lambda s: int(s["T_star"]))
        n = len(ranked)
        third = max(1, n // 3)

        early = ranked[:third]
        late = ranked[-third:]

        early_rates.append(_error_rate(early))
        late_rates.append(_error_rate(late))

    x = list(range(len(datasets)))
    width = 0.38
    early_x = [xi - width / 2 for xi in x]
    late_x = [xi + width / 2 for xi in x]

    plt.figure(figsize=(8, 5))
    plt.bar(early_x, early_rates, width=width, label="Early T* (bottom 33%)")
    plt.bar(late_x, late_rates, width=width, label="Late T* (top 33%)")

    plt.xticks(x, datasets)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Dataset")
    plt.ylabel("Error Rate")
    plt.title("Premature Disengagement vs Error Rate")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    out_path = figures_dir / "t_star_error_correlation.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(samples: list[dict], output_dir: str | Path = "figures") -> None:
    """Generate and save all required causal analysis plots."""
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_samples_by_dataset(samples)
    max_step = _max_step_from_samples(samples)

    _plot_causal_decay_curves(grouped, figures_dir=figures_dir)
    _plot_t_star_histograms(grouped, max_step=max_step, figures_dir=figures_dir)
    _plot_t_star_error_correlation(grouped, figures_dir=figures_dir)
