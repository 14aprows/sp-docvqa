import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.grid_search import plot_loss, save_results

def run_script(script_name, *arguments):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script_name),
            *map(str, arguments),
        ],
        cwd=ROOT,
        check=True,
    )

def main():
    config_path = ROOT / "configs" / "layoutlmv3_full.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        base_config = yaml.safe_load(file)

    learning_rates = base_config["grid_search"]["step1_lr"]["learning_rate"]

    for name in ("train_path", "val_path", "val_eval_path"):
        data_path = ROOT / base_config["data"][name]
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found: {data_path}")

    output_dir = ROOT / "outputs" / "grid_search" / "step1_lr"
    best_model_dir = output_dir / "best_model"
    if best_model_dir.exists():
        raise FileExistsError(f"Best model folder already exists: {best_model_dir}")
    for index in range(1, len(learning_rates) + 1):
        trial_dir = output_dir / f"lr_{index}"
        if trial_dir.exists():
            raise FileExistsError(f"Trial folder already exists: {trial_dir}")
    rows = []

    for index, learning_rate in enumerate(learning_rates, start=1):
        trial_name = f"lr_{index}"
        trial_dir = output_dir / trial_name

        trial_dir.mkdir(parents=True)
        trial_config = copy.deepcopy(base_config)
        trial_config.pop("grid_search", None)

        trial_config["training"]["backbone_learning_rate"] = learning_rate
        trial_config["training"]["new_layers_learning_rate"] = learning_rate
        trial_config["training"]["resume_from"] = None
        trial_config["training"]["output_dir"] = str(trial_dir / "checkpoints")
        trial_config["evaluation"]["checkpoint_path"] = str(trial_dir / "checkpoints" / "best")
        trial_config["evaluation"]["output_dir"] = str(trial_dir / "evaluation")

        generated_config_path = trial_dir / "config.yaml"
        with generated_config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(trial_config, file, sort_keys=False, allow_unicode=True)
        print(f"\n{trial_name}: learning rate = {learning_rate}")

        run_script("train.py", "--config", generated_config_path)
        run_script("evaluate.py", "--config", generated_config_path)

        history_path = trial_dir / "checkpoints" / "last" / "training_history.json"
        best_epoch = plot_loss(history_path, trial_dir / "learning_curve.png")
        shutil.copy2(history_path, trial_dir / "training_history.json")
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        summary_path = trial_dir / "evaluation" / "evaluation_summary.json"
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        row = {
            "trial": trial_name,
            "learning_rate": learning_rate,
            "best_epoch": best_epoch["epoch"],
            "best_train_loss": best_epoch["train_loss"],
            "best_validation_loss": best_epoch["validation_loss"],
            "last_epoch": history[-1]["epoch"],
            "last_train_loss": history[-1]["train_loss"],
            "last_validation_loss": history[-1]["validation_loss"],
        }
        for section in ("data", "preprocessing", "model", "training", "inference"):
            for name, value in trial_config[section].items():
                row[f"{section}_{name}"] = value
        row.update(summary["overall"])
        rows.append(row)

        save_results(rows, output_dir / "results.csv")

    best = max(rows, key=lambda row: (row["anls"], -row["best_validation_loss"]))
    best_trial_dir = output_dir / best["trial"]
    best_dir = best_trial_dir / "checkpoints" / "best"
    shutil.copytree(best_dir, best_model_dir)
    best["checkpoint_path"] = str(best_model_dir)
    with (output_dir / "best.json").open("w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print("\nBest learning rate by ANLS:", best)

    evaluation_dir = best_trial_dir / "evaluation"
    run_script(
        "analyze_errors.py",
        "--predictions", evaluation_dir / "predictions_per_question.json",
        "--output-dir", evaluation_dir,
    )
    run_script(
        "visualize.py",
        "--predictions", evaluation_dir / "error_cases.json",
        "--output-dir", best_trial_dir / "visualizations",
        "--limit", 20,
    )

    output_root = output_dir.resolve(strict=True)
    for index in range(1, len(learning_rates) + 1):
        trial_dir = (output_dir / f"lr_{index}").resolve(strict=True)
        checkpoint_dir = (trial_dir / "checkpoints").resolve(strict=True)
        if checkpoint_dir.parent != trial_dir or not checkpoint_dir.is_relative_to(output_root):
            raise ValueError(f"Checkpoint folder is outside the expected path: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    main()