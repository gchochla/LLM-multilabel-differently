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
        "--dataset", type=str, help="Dataset name.", required=True
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
        nargs="+",
        help="Maximum rank to plot.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not show legend.",
    )
    parser.add_argument(
        "--no-boxplot",
        action="store_true",
        help="Do not show boxplot.",
    )
    parser.add_argument(
        "--no-violin",
        action="store_true",
        help="Do not show violin plot.",
    )
    parser.add_argument(
        "--no-xtitle",
        action="store_true",
        help="Do not show x title.",
    )

    args = parser.parse_args()
    assert len(args.experiment) == len(args.name), (
        "Number of experiments and names must match. "
        f"Got {len(args.experiment)} experiments and {len(args.name)} names."
    )

    if args.max_rank is not None:
        if len(args.max_rank) == 1:
            args.max_rank = args.max_rank * len(args.experiment)
        assert len(args.max_rank) == len(args.experiment), (
            "Number of max ranks must match number of experiments. "
            f"Got {len(args.max_rank)} max ranks and {len(args.experiment)} experiments."
        )
    else:
        args.max_rank = [None] * len(args.experiment)

    assert not (
        args.no_violin and args.no_boxplot
    ), "Cannot disable both violin and boxplot. Please choose at least one."
    return args


def entropy(p):
    """Calculate the entropy of a probability distribution."""
    p = np.array(p)
    p = p[p > 0]  # Ignore zero probabilities
    return -np.sum(p * np.log2(p))


def plot_violin_and_box(
    axis, data_list, positions, color, hatch, width, boxplot=True, violin=True
):
    """Draw a violin and an inset boxplot for a single series."""
    # Skip positions where data is empty
    data_and_pos = [(d, p) for d, p in zip(data_list, positions) if len(d) > 0]
    if not data_and_pos:
        return
    filtered_data, filtered_pos = zip(*data_and_pos)

    if violin:
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
    if boxplot:
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


def main():
    args = parse_args()

    experiment_stats = {}

    width = 0.20
    second_width = 0.10

    second_legend_labels = [
        "last",
        "intermediate",
        "$r$+1 pred",
        "intermediate @ $r$+1",
        # "$r$+1 pred @ $r$+1",
    ]
    ent_legend_labels = [
        "last",
        "intermediate",
    ]
    prob_legend_labels = [
        "last",
        "intermediate",
    ]

    os.makedirs(args.output, exist_ok=True)

    print(f"====== {args.dataset} ======")

    for experiment, name, max_rank in zip(
        args.experiment, args.name, args.max_rank
    ):
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

        fn_name = name.lower().replace(" ", "_")
        with open(
            os.path.join(args.output, f"{fn_name}_stats.yml"),
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(
                {
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

        if max_rank is not None:
            # < because user sees rank +1 in plot
            ranks = [r for r in ranks if r < max_rank]

        experiment_stats[name] = {
            "stats_as_last_label": stats_as_last_label,
            "stats_as_intermediate_label": stats_as_intermediate_label,
            "avg_stats_as_last_label": avg_stats_as_last_label,
            "avg_stats_as_intermediate_label": avg_stats_as_intermediate_label,
            "ranks": ranks,
        }

    max_rank = (
        max(max(stats["ranks"]) for stats in experiment_stats.values()) + 1
    )

    # ---------------- Probabilities subplot figure ----------------
    fig_prob, ax2 = plt.subplots(
        nrows=1,
        ncols=len(experiment_stats),
        figsize=(max_rank * len(experiment_stats) * 2, 3),
        sharey=True,
    )
    fig_prob_top, ax1 = plt.subplots(
        nrows=1,
        ncols=len(experiment_stats),
        figsize=(max_rank * len(experiment_stats) * 2, 3),
        sharey=True,
    )

    fig_ent, ax_ent = plt.subplots(
        figsize=(max_rank * len(experiment_stats) * 2, 3),
        nrows=1,
        ncols=len(experiment_stats),
        sharey=True,
    )

    for i, (name, stats) in enumerate(experiment_stats.items()):
        ranks = stats["ranks"]

        non_last_tops = [
            stats["stats_as_intermediate_label"].get(r, {}).get("top", [])
            for r in ranks
        ]
        last_tops = [
            stats["stats_as_last_label"].get(r, {}).get("top", [])
            for r in ranks
        ]
        # include entropy from *both* intermediate and last for every rank
        non_last_entropies = [
            stats["stats_as_intermediate_label"].get(r, {}).get("entropy", [])
            for r in ranks
        ]
        last_entropies = [
            stats["stats_as_last_label"].get(r, {}).get("entropy", [])
            for r in ranks
        ]

        base_positions = np.arange(len(ranks))

        prob_series = [
            (last_tops, -0.15, 'tab:orange', '\\\\'),
            (non_last_tops, 0.15, 'tab:blue', '//'),
        ]
        ent_series = [
            (last_entropies, -0.15, 'tab:orange', '\\\\'),
            (non_last_entropies, 0.15, 'tab:blue', '//'),
        ]

        # Bottom: second highest probabilities
        second_non_last = [
            stats["stats_as_intermediate_label"]
            .get(r, {})
            .get("expected_next_label_scores", [])
            for r in ranks
        ]
        actual_second_non_last = [
            stats["stats_as_intermediate_label"]
            .get(r, {})
            .get("actual_next_label_scores", [])
            for r in ranks
        ]
        real_second_non_last = [
            stats["stats_as_intermediate_label"]
            .get(r, {})
            .get("expected_next_label_scores_p1", [])
            for r in ranks
        ]
        real_actual_second_non_last = [
            stats["stats_as_intermediate_label"]
            .get(r, {})
            .get("actual_next_label_scores_p1", [])
            for r in ranks
        ]
        second_last = [
            stats["stats_as_last_label"].get(r, {}).get("second", [])
            for r in ranks
        ]
        second_series = [
            (second_last, -0.225, 'tab:orange', '\\\\'),
            (second_non_last, -0.075, 'tab:blue', '//'),
            (actual_second_non_last, 0.075, 'tab:green', 'x'),
            (real_second_non_last, 0.225, 'tab:gray', '-'),
            # (real_actual_second_non_last, 0.30, 'tab:red', 'o'),
        ]

        # Top: existing probability plot
        for data, offset, color, hatch in prob_series:
            plot_violin_and_box(
                ax1[i],
                data,
                base_positions + offset,
                color,
                hatch,
                width,
                boxplot=not args.no_boxplot,
                violin=not args.no_violin,
            )
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

        for data, offset, color, hatch in second_series:
            plot_violin_and_box(
                ax2[i],
                data,
                base_positions + offset,
                color,
                hatch,
                second_width,
                boxplot=not args.no_boxplot,
                violin=not args.no_violin,
            )

        # Add accuracy lines connecting mid and r+1 pred using average is_next_pred_correct
        accuracy_means = [
            stats["avg_stats_as_intermediate_label"]
            .get(r, {})
            .get("is_next_pred_correct", (None, None))[0]
            for r in ranks
        ]
        for idx, acc in enumerate(accuracy_means):
            if acc is not None:
                x_mid = base_positions[idx] - 0.075
                x_pred = base_positions[idx] + 0.075
                # Determine y position using min of mid and r+1 pred second-series values
                mid_vals = second_non_last[idx]
                pred_vals = actual_second_non_last[idx]
                y = min(
                    min(mid_vals) if mid_vals else 0,
                    min(pred_vals) if pred_vals else 0,
                )
                ax2[i].annotate(
                    "",
                    xy=(x_pred, y),
                    xytext=(x_mid, y),
                    arrowprops=dict(
                        arrowstyle="<->",
                        linestyle="-",
                        linewidth=1.5,
                        color="black",
                        connectionstyle="arc3,rad=0.2",
                    ),
                )
                ax2[i].text(
                    (x_mid + x_pred) / 2,
                    y - 0.14,
                    f"same: {acc * 100:.1f}%",
                    va="bottom",
                    ha="center",
                    fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.9),
                )

        ax2[i].set_xticks(base_positions)
        ax2[i].set_xticklabels([None] * len(ranks))
        ax1[i].set_xticks(base_positions)
        # for lower ranks, clear the tick labels
        ax1[i].set_xticklabels([None] * len(ranks))
        ax1[i].set_xticklabels(np.array(ranks) + 1, fontsize=16)
        ax1[i].set_title(name, fontsize=18)
        ax2[i].set_title(name, fontsize=18)
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

        fn_name = name.lower().replace(" ", "_")

        plt.close(fig_prob)

        # ---------------- Entropy figure ----------------

        for data, offset, color, hatch in ent_series:
            plot_violin_and_box(
                ax_ent[i],
                data,
                base_positions + offset,
                color,
                hatch,
                width,
                boxplot=not args.no_boxplot,
                violin=not args.no_violin,
            )

        # X‑axis formatting
        ax_ent[i].set_xticks(base_positions)
        ax_ent[i].set_xticklabels(np.array(ranks) + 1)

        # Y‑axis / title / legend
        if i == 0:
            ax_ent[i].set_ylabel("Entropy (bits)", fontsize=16)
        ax_ent[i].set_title(name, fontsize=18)
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
        # if i == 0:
        #     ax_ent[i].legend(
        #         handles=ent_legend_handles, loc='upper left', fontsize=12
        #     )

        ax2[i].set_xlim(
            second_series[0][1] - second_width - 0.02,
            ranks[-1] + second_series[-1][1] + second_width + 0.02,
        )
        ax1[i].set_xlim(
            second_series[0][1] - second_width - 0.02,
            ranks[-1] + second_series[-1][1] + second_width + 0.02,
        )
        ax_ent[i].set_xlim(
            second_series[0][1] - second_width - 0.02,
            ranks[-1] + second_series[-1][1] + second_width + 0.02,
        )

    ncol = len(second_legend_labels) // 2
    bbox = (0.03, 1.30)
    ncol_top = len(prob_legend_labels)
    bbox_top = (0.03, 1.08)

    if not args.no_legend:
        fig_prob.legend(
            handles=second_legend_handles,
            loc="upper left",
            fontsize=18,
            bbox_to_anchor=bbox,
            ncol=ncol,
        )
        fig_prob_top.legend(
            handles=prob_legend_handles,
            loc="upper left",
            fontsize=18,
            bbox_to_anchor=bbox_top,
            ncol=ncol_top,
        )

    fig_prob.tight_layout()

    fig_prob.suptitle(
        f"Second-highest Probabilities for {args.dataset}",
        fontsize=20,
        y=1.09,
    )
    if not args.no_xtitle:
        # add x axis label
        fig_prob.text(
            0.515,
            -0.02,
            "Prediction Step $r$",
            ha="center",
            va="center",
            fontsize=20,
        )

    fig_prob.savefig(
        os.path.join(args.output, "second_probabilities.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    fig_prob.savefig(
        os.path.join(args.output, "second_probabilities.png"),
        bbox_inches="tight",
        dpi=300,
    )

    fig_prob_top.suptitle(
        f"Top Probabilities for {args.dataset}", fontsize=18, y=0.96
    )
    fig_prob_top.tight_layout()
    if not args.no_xtitle:
        # add x axis label
        fig_prob_top.text(
            0.515,
            0.00,
            "Prediction Step $r$",
            ha="center",
            va="center",
            fontsize=20,
        )

    fig_prob_top.savefig(
        os.path.join(args.output, "only_top_probabilities.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    fig_prob_top.savefig(
        os.path.join(args.output, "only_top_probabilities.png"),
        bbox_inches="tight",
        dpi=300,
    )

    if not args.no_legend:
        fig_ent.legend(
            handles=ent_legend_handles,
            loc="upper left",
            fontsize=17,
            bbox_to_anchor=bbox,
            ncol=ncol,
        )
    fig_ent.suptitle(
        f"Entropy for {args.dataset}",
        fontsize=20,
        y=0.98,
    )
    if not args.no_xtitle:
        fig_ent.text(
            0.51,
            0.00,
            "Prediction Step $r$",
            ha="center",
            va="center",
            fontsize=20,
        )
    fig_ent.tight_layout()
    fig_ent.savefig(
        os.path.join(args.output, "entropy.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    fig_ent.savefig(
        os.path.join(args.output, "entropy.png"),
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    main()
