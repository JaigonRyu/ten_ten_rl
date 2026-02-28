import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def get_model_name(name: str) -> str:
    if "greedy" in name:
        name = "Greedy"
    if "random" in name:
        name = "Random"
    if "ppo" in name:
        name = "PPO"
    if "dqn" in name:
        name = "DQN"
    return name

def summarize_scores(scores: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "count": int(scores.shape[0]),
    }


def plot_scores(name: str, scores: np.ndarray, out_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=100, alpha=0.8, edgecolor="black")
    plt.title(f"{name} Score distribution for {len(scores)} episodes")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"{name}_scores.png")
    plt.savefig(outfile)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize run_game results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ten_ten_rl/results",
        help="Directory containing *_scores.npy and *_times.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ten_ten_rl/visuals",
        help="Directory to write summary plots (defaults to results dir)",
    )
    parser.add_argument(
        "--aggregate-adversarial",
        action="store_true",
        help="If set, aggregate all adversarial=True score files into one plot/stat",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    score_files = sorted(glob.glob(os.path.join(args.results_dir, "*_scores*.npy")))
    time_files = sorted(glob.glob(os.path.join(args.results_dir, "*_elapsed*.npy")))

    if not score_files:
        print(f"No *_scores*.npy found in {args.results_dir}")
        return

    adversarial_scores = []
    adversarial_names = []
    adversarial_by_model = {}

    print("=== Score summaries ===")
    for sf in score_files:
        name = os.path.splitext(os.path.basename(sf))[0]
        scores = np.load(sf)

        if args.aggregate_adversarial and "True" in name:
            adversarial_scores.append(scores)
            adversarial_names.append(name)
            tokens = name.split("_")
            model = next(
                (t for t in tokens if t in {"random", "greedy", "dqn", "ppo"}),
                tokens[0],
            )
            adversarial_by_model.setdefault(model, []).append(scores)
            continue
        if "True" in name:
            continue

        name = get_model_name(name)

        stats = summarize_scores(scores)
        print(
            f"{name}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
            f"min={stats['min']:.2f} max={stats['max']:.2f} n={stats['count']}"
        )
        plot_scores(name, scores, args.out_dir)

    if args.aggregate_adversarial and adversarial_scores:
        combined = np.concatenate(adversarial_scores, axis=0)
        stats = summarize_scores(combined)
        agg_name = "adversarial_all"
        print(
            f"{agg_name}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
            f"min={stats['min']:.2f} max={stats['max']:.2f} n={stats['count']}"
        )
        plot_scores(agg_name, combined, args.out_dir)

        # Bar chart of mean scores by model (with std error bars)
        models = sorted(adversarial_by_model.keys())
        data = [np.concatenate(adversarial_by_model[m], axis=0) for m in models]
        models = [get_model_name(m) for m in models]
        means = [np.mean(d) for d in data]
        stds = [np.std(d) for d in data]
        plt.figure(figsize=(10, 6))
        plt.bar(
            models,
            means,
            yerr=stds,
            capsize=5,
            edgecolor="black",
            alpha=0.85,
        )
        plt.ylabel("Score")
        plt.xlabel("Model")
        plt.title("Scores Against Minimax Adversary")
        plt.tight_layout()
        outfile = os.path.join(args.out_dir, "adversarial_by_model.png")
        plt.savefig(outfile)
        plt.close()

    # if time_files:
    #     print("\n=== Times ===")
    #     for tf in time_files:
    #         name = os.path.splitext(os.path.basename(tf))[0]
    #         times = np.load(tf)
    #         # times may be a scalar total or array of per-episode durations
    #         if np.isscalar(times) or times.shape == ():
    #             total = float(times)
    #             print(f"{name}: total={total:.2f}s")
    #         else:
    #             print(
    #                 f"{name}: mean={np.mean(times):.3f}s std={np.std(times):.3f}s "
    #                 f"min={np.min(times):.3f}s max={np.max(times):.3f}s n={times.shape[0]}"
    #             )


if __name__ == "__main__":
    main()
