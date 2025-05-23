import yaml
import os

import gridparse
import numpy as np


def parse_args():
    parser = gridparse.ArgumentParser(
        description="Calculate attention weights between labels and input from a YAML file."
    )
    parser.add_argument(
        "--experiment", type=str, nargs="+", help="Path to experiment folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="+",
        help="Output directory for the plots.",
        default="plots",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for f, out in zip(args.experiment, args.output):
        name = os.path.basename(os.path.abspath(f))
        print(f"==={f}===")

        # Load the YAML file
        yaml_file = os.path.join(f, "indexed_metrics.yml")
        with open(yaml_file) as file:
            metrics = yaml.safe_load(file)

        metrics = {
            (k + "_" + kk): vv
            for k, v in metrics.items()
            for kk, vv in v.items()
            if kk != "description"
        }

        avg_inp_attn = []
        avg_label_attn = []
        avg_label_token_attn = []
        for v in metrics.values():
            if "test_label_v_input_attn" not in v:
                continue
            attn = v["test_label_v_input_attn"]
            avg_inp_attn.extend([a / n for a, n in attn["input"]])
            avg_label_attn.extend([a / n for a, n in attn["prev_label"]])
            avg_label_token_attn.extend(
                [a / n for a, n in attn["only_label_token"]]
            )

        avg_inp_attn = np.array(avg_inp_attn).mean()
        avg_label_attn = np.array(avg_label_attn).mean()
        avg_label_token_attn = np.array(avg_label_token_attn).mean()

        s = f"Avg input attention: {avg_inp_attn}\n"
        s += f"Avg label attention: {avg_label_attn}\n"
        s += f"Avg label token attention: {avg_label_token_attn}\n"
        print(f"Avg input attention: {avg_inp_attn}")
        print(f"Avg label attention: {avg_label_attn}")
        print(f"Avg label token attention: {avg_label_token_attn}")

        # Save the results to a file
        os.makedirs(out, exist_ok=True)
        out_file = os.path.join(out, f"{name}_attn_weights.txt")
        with open(out_file, "w") as f:
            f.write(f"Avg input attention: {avg_inp_attn}\n")
            f.write(f"Avg label attention: {avg_label_attn}\n")
            f.write(f"Avg label token attention: {avg_label_token_attn}\n")


if __name__ == "__main__":
    main()
