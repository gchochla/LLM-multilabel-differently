import os
import yaml

import gridparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def parse_args():
    parser = gridparse.ArgumentParser(
        description="Plot distributions of scores."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        help="Experiments to plot.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for the plots.",
        default="plots",
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="+",
        help="Name of the plot.",
    )
    parser.add_argument(
        "--remove-none",
        action="store_true",
        help="Remove None values from the scores.",
    )
    parser.add_argument(
        "--union",
        action="store_true",
        help="Show ranks without next preds.",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        help="Maximum rank to plot.",
    )

    args = parser.parse_args()
    assert len(args.experiment) == len(args.name), (
        "Number of experiments and names must match."
        f"Got {len(args.experiment)} experiments and {len(args.name)} names."
    )
    return args


def entropy(p):
    """Calculate the entropy of a probability distribution."""
    p = np.array(p)
    p = p[p > 0]  # Ignore zero probabilities
    return -np.sum(p * np.log2(p))


def main():
    args = parse_args()

    for experiment, name in zip(args.experiment, args.name):
        print(f"=== {name} ===")

        stats_as_last_label = {}
        avg_stats_as_last_label = {}
        stats_as_intermediate_label = {}
        avg_stats_as_intermediate_label = {}
        second_not_predicted_ever = 0
        cnt_for_second_not_predicted_ever = 0

        metrics_file = os.path.join(experiment, "indexed_metrics.yml")
        with open(metrics_file) as fp:
            metrics = yaml.safe_load(fp)

        metrics = {
            (k + "_" + kk): vv
            for k, v in metrics.items()
            for kk, vv in v.items()
            if kk != "description"
        }

        for k, v in metrics.items():
            scores: list[dict[str, float]] = v["test_scores"]
            preds = v["test_preds"]
            i = 0
            while i < len(scores):
                ith_label_scores = scores[i]
                if args.remove_none and max(ith_label_scores.values()) in (
                    ith_label_scores["none"],
                    ith_label_scores.get("nothing", -1),
                ):
                    print(k)
                    break

                if i == len(scores) - 1:
                    sorted_scores = sorted(
                        ith_label_scores.values(), reverse=True
                    )
                    new_dict = dict(
                        top=sorted_scores[0],
                        ratio=sorted_scores[0] / sorted_scores[1],
                        second=sorted_scores[1],
                        entropy=entropy(sorted_scores),
                    )
                    for kk, vv in new_dict.items():
                        stats_as_last_label.setdefault(i, {}).setdefault(
                            kk, []
                        ).append(vv)
                else:
                    ip1th_label_scores = scores[i + 1]
                    sorted_scores = sorted(
                        ith_label_scores.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    # see if expected next label based on current probabilities
                    # is the same as the actual next label
                    expected_next_label = sorted_scores[1][0]
                    # see the actual next label, what probability it has in the next label
                    sorted_next_scores = sorted(
                        ip1th_label_scores.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    actual_next_label = sorted_next_scores[0][0]
                    current_label = sorted_scores[0][0]
                    # if the current label is the same as the expected next label
                    # then don't count the actual next label' score

                    is_next_pred_correct = (
                        expected_next_label == actual_next_label
                    )
                    if not is_next_pred_correct:
                        cnt_for_second_not_predicted_ever += 1
                        second_not_predicted_ever += int(
                            expected_next_label not in preds
                        )

                    new_dict = dict(
                        top=sorted_scores[0][1],
                        ratio=sorted_scores[0][1] / sorted_scores[1][1],
                        entropy=entropy([x[1] for x in sorted_scores]),
                        expected_next_label_scores=sorted_scores[1][1],
                        expected_next_label_scores_p1=ip1th_label_scores[
                            expected_next_label
                        ],
                        actual_next_label_scores_p1=sorted_next_scores[0][1],
                        is_next_pred_correct=is_next_pred_correct,
                    )
                    if actual_next_label != current_label:
                        # we do this to avoid peaks in this distribution
                        new_dict["actual_next_label_scores"] = ith_label_scores[
                            actual_next_label
                        ]

                    for kk, vv in new_dict.items():
                        stats_as_intermediate_label.setdefault(
                            i, {}
                        ).setdefault(kk, []).append(vv)
                i += 1

        # average the stats and store in avg
        for k, v in stats_as_last_label.items():
            for kk, vv in v.items():
                avg_stats_as_last_label.setdefault(k, {})[kk] = (
                    np.mean(vv),
                    np.std(vv),
                )
        for k, v in stats_as_intermediate_label.items():
            for kk, vv in v.items():
                avg_stats_as_intermediate_label.setdefault(k, {})[kk] = (
                    np.mean(vv),
                    np.std(vv),
                )

        # print the stats
        # and store them in the experiment folder
        s = f"avg stats as last label: {avg_stats_as_last_label}"
        s += f"\n\navg stats as intermediate label: {avg_stats_as_intermediate_label}"
        if cnt_for_second_not_predicted_ever > 0:
            s += f"\n\nsecond not predicted ever: {second_not_predicted_ever / cnt_for_second_not_predicted_ever * 100:.1f}%"
        print(s)
        with open(
            os.path.join(experiment, "stats.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(s)

        # ---- violin plot ----
        # Collect the per‑rank distributions
        if args.union:
            ranks = sorted(
                set(stats_as_intermediate_label.keys()).union(
                    set(stats_as_last_label.keys())
                )
            )
        else:
            ranks = sorted(
                set(stats_as_intermediate_label.keys()).intersection(
                    set(stats_as_last_label.keys())
                )
            )

        if args.max_rank is not None:
            # < because user sees rank +1 in plot
            ranks = [r for r in ranks if r < args.max_rank]

        non_last_tops = [
            stats_as_intermediate_label.get(r, {}).get("top", []) for r in ranks
        ]
        last_tops = [
            stats_as_last_label.get(r, {}).get("top", []) for r in ranks
        ]
        # include entropy from *both* intermediate and last for every rank
        non_last_entropies = [
            stats_as_intermediate_label.get(r, {}).get("entropy", [])
            for r in ranks
        ]
        last_entropies = [
            stats_as_last_label.get(r, {}).get("entropy", []) for r in ranks
        ]

        base_positions = np.arange(len(ranks))

        prob_series = [
            (last_tops, -0.15, 'tab:orange', '\\\\'),
            (non_last_tops, 0.15, 'tab:blue', '//'),
        ]
        prob_legend_labels = [
            "last",
            "mid",
        ]
        ent_series = [
            (last_entropies, -0.15, 'tab:orange', '\\\\'),
            (non_last_entropies, 0.15, 'tab:blue', '//'),
        ]
        ent_legend_labels = [
            "last",
            "mid",
        ]

        # Bottom: second highest probabilities
        second_non_last = [
            stats_as_intermediate_label.get(r, {}).get(
                "expected_next_label_scores", []
            )
            for r in ranks
        ]
        actual_second_non_last = [
            stats_as_intermediate_label.get(r, {}).get(
                "actual_next_label_scores", []
            )
            for r in ranks
        ]
        real_second_non_last = [
            stats_as_intermediate_label.get(r, {}).get(
                "expected_next_label_scores_p1", []
            )
            for r in ranks
        ]
        real_actual_second_non_last = [
            stats_as_intermediate_label.get(r, {}).get(
                "actual_next_label_scores_p1", []
            )
            for r in ranks
        ]
        second_last = [
            stats_as_last_label.get(r, {}).get("second", []) for r in ranks
        ]
        second_series = [
            (second_last, -0.30, 'tab:orange', '\\\\'),
            (second_non_last, -0.15, 'tab:blue', '//'),
            (actual_second_non_last, 0, 'tab:green', 'x'),
            (real_second_non_last, 0.15, 'tab:gray', '-'),
            (real_actual_second_non_last, 0.30, 'tab:red', 'o'),
        ]
        second_legend_labels = [
            "last",
            "mid",
            "$r$+1 pred",
            "mid @ $r$+1",
            "$r$+1 pred\n@ $r$+1",
        ]

        width = 0.20
        second_width = 0.10

        def plot_violin_and_box(
            axis, data_list, positions, color, hatch, width=width
        ):
            """Draw a violin and an inset boxplot for a single series."""
            # Skip positions where data is empty
            data_and_pos = [
                (d, p) for d, p in zip(data_list, positions) if len(d) > 0
            ]
            if not data_and_pos:
                return
            filtered_data, filtered_pos = zip(*data_and_pos)

            vp = axis.violinplot(
                filtered_data,
                positions=filtered_pos,
                widths=width,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in vp['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor('black')
                body.set_alpha(0.5)
                body.set_hatch(hatch)

            # Overlay a boxplot on top of each violin
            bp = axis.boxplot(
                filtered_data,
                positions=filtered_pos,
                widths=width / 2,
                vert=True,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_alpha(0.75)
                patch.set_hatch(hatch)
            for element in ('whiskers', 'caps', 'medians'):
                for line in bp[element]:
                    line.set_color('black')

        # ---------------- Probabilities subplot figure ----------------
        fig_prob, (ax2, ax1) = plt.subplots(
            2, 1, figsize=(max(3, len(ranks)) * 3, 6)
        )
        # Top: existing probability plot
        for data, offset, color, hatch in prob_series:
            plot_violin_and_box(
                ax1, data, base_positions + offset, color, hatch
            )
        ax1.set_xticks(base_positions)
        ax1.set_xticklabels(np.array(ranks) + 1, fontsize=16)
        ax1.set_title(f"Highest probabilities", fontsize=16)
        prob_legend_handles = [
            Patch(
                facecolor=rest[2],
                edgecolor='black',
                hatch=rest[3],
                label=label,
                alpha=0.7,
            )
            for label, rest in zip(prob_legend_labels, prob_series)
        ]
        ax1.legend(handles=prob_legend_handles, loc='upper left', fontsize=12)

        for data, offset, color, hatch in second_series:
            plot_violin_and_box(
                ax2,
                data,
                base_positions + offset,
                color,
                hatch,
                width=second_width,
            )

        # Add accuracy lines connecting mid and r+1 pred using average is_next_pred_correct
        accuracy_means = [
            avg_stats_as_intermediate_label.get(r, {}).get(
                "is_next_pred_correct", (None, None)
            )[0]
            for r in ranks
        ]
        for idx, acc in enumerate(accuracy_means):
            if acc is not None:
                x_mid = base_positions[idx] - 0.15
                x_pred = base_positions[idx]
                # Determine y position using min of mid and r+1 pred second-series values
                mid_vals = second_non_last[idx]
                pred_vals = actual_second_non_last[idx]
                y = min(
                    min(mid_vals) if mid_vals else 0,
                    min(pred_vals) if pred_vals else 0,
                )
                ax2.annotate(
                    "",
                    xy=(x_pred, y),
                    xytext=(x_mid, y),
                    arrowprops=dict(
                        arrowstyle="<->",
                        linestyle="-",
                        linewidth=1,
                        color="black",
                        connectionstyle="arc3,rad=0.2",
                    ),
                )
                ax2.text(
                    (x_mid + x_pred) / 2,
                    y - 0.13,
                    f"same: {acc * 100:.1f}%",
                    va="bottom",
                    ha="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.9),
                )

        ax2.set_xticks(base_positions)
        ax2.set_xticklabels([None] * len(ranks))
        ax1.set_xlabel("Prediction Index $r$", fontsize=16)
        ax2.set_title(f"Second highest probabilities", fontsize=16)
        second_legend_handles = [
            Patch(
                facecolor=rest[2],
                edgecolor='black',
                hatch=rest[3],
                label=label,
                alpha=0.7,
            )
            for label, rest in zip(second_legend_labels, second_series)
        ]
        ax2.legend(handles=second_legend_handles, loc='upper left', fontsize=10)
        fig_prob.suptitle(f"Probabilities for {name}", fontsize=18)

        fn_name = name.lower().replace(" ", "_")

        plt.tight_layout()
        os.makedirs(args.output, exist_ok=True)
        plt.savefig(
            os.path.join(args.output, f"{fn_name}_top_probabilities.pdf"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            os.path.join(args.output, f"{fn_name}_top_probabilities.png"),
            bbox_inches="tight",
            dpi=300,
        )

        plt.close(fig_prob)

        # ---------------- Entropy figure ----------------
        fig_ent, ax_ent = plt.subplots(figsize=(max(3, len(ranks)) * 3, 3))

        for data, offset, color, hatch in ent_series:
            plot_violin_and_box(
                ax_ent, data, base_positions + offset, color, hatch
            )

        # X‑axis formatting
        ax_ent.set_xticks(base_positions)
        ax_ent.set_xticklabels(np.array(ranks) + 1)
        ax_ent.set_xlabel("Prediction Index $r$", fontsize=16)

        # Y‑axis / title / legend
        ax_ent.set_ylabel("Entropy (bits)", fontsize=16)
        ax_ent.set_title(f"Entropies for {name}", fontsize=18)
        ent_legend_handles = [
            Patch(
                facecolor=rest[2],
                edgecolor='black',
                hatch=rest[3],
                label=label,
                alpha=0.7,
            )
            for label, rest in zip(ent_legend_labels, ent_series)
        ]
        ax_ent.legend(handles=ent_legend_handles, loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output, f"{fn_name}_entropy.pdf"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            os.path.join(args.output, f"{fn_name}_entropy.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig_ent)

        with open(
            os.path.join(args.output, f"{fn_name}_stats.yml"),
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(
                {
                    "avg_stats_as_last_label": avg_stats_as_last_label,
                    "avg_stats_as_intermediate_label": avg_stats_as_intermediate_label,
                    "second_not_predicted_ever": (
                        (
                            second_not_predicted_ever
                            / cnt_for_second_not_predicted_ever
                            * 100
                        )
                        if cnt_for_second_not_predicted_ever > 0
                        else None
                    ),
                },
                f,
                default_flow_style=False,
            )


if __name__ == "__main__":
    main()
