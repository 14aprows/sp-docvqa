import copy
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.grid_search import plot_loss, save_results

def run_script(name, config_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--config", str(config_path)],
        cwd=ROOT,
        check=True
    )

def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_config(config, path):
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

def main():
    config_path = ROOT / "configs" / "layoutlmv3_full.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        base_config = yaml.safe_load(file)

    grid = base_config.pop("grid_search")
    parameters = [("training", "learning_rate", grid["learning_rate"])]

    for section in ("preprocessing", "model", "training"):
        for name, values in grid[section].items():
            parameters.append((section, name, values))

    combinations = list(product(*(values for _, _, values in parameters)))
    if not combinations:
        raise ValueError("Each grad list needs at least one value")

    use_oca = base_config["model"].get("use_oca", True)
    use_mcs = base_config["model"].get("use_mcs", True)

    config_names = {
        (False, False): "baseline",
        (True, False): "oca_only",
        (False, True): "mcs_only",
        (True, True): "oca_mcs"
    }

    config_name = config_names[(use_oca, use_mcs)]
    output_dir = ROOT / "outputs" / "grid_search" / config_name
    output_dir.mkdir(parents=True)
    best_dir = output_dir / "best_model"

    prepared_data = {}
    rows = []
    best_score = None

    print(f"Total trials: {len(combinations)}")

    for index, combination in enumerate(combinations, start=1):
        trial_dir = output_dir / f"trial_{index:03d}"
        trial_dir.mkdir()
        trial_config = copy.deepcopy(base_config)
        row = {"trial": trial_dir.name}

        for (section, name, _), value in zip(parameters, combination):
            if name == "learning_rate":
                trial_config["training"]["backbone_learning_rate"] = value
                trial_config["training"]["new_layers_learning_rate"] = value
            elif name == "window_pairs":
                trial_config["preprocessing"]["max_words"] = value[0]
                trial_config["preprocessing"]["stride"] = value[1]
            elif name == "oca_size_pairs":
                trial_config["model"]["adapter_size"] = value[0]
                trial_config["model"]["confidence_size"] = value[1]
            else:
                trial_config[section][name] = value      

        trial_config["training"]["epochs"] = grid["epochs"]
        trial_config["training"]["early_stopping_patience"] = grid["patience"]
        trial_config["training"]["resume_from"] = None

        data_key = tuple(sorted(trial_config["preprocessing"].items()))
        needs_preprocessing = data_key not in prepared_data
        if needs_preprocessing:
            number = len(prepared_data) + 1
            prefix = f"data/processed/{config_name}_{number:02d}"
            prepared_data[data_key] = {
                "train_path": f"{prefix}_train.json",
                "val_path": f"{prefix}_val.json",
                "val_eval_path": f"{prefix}_val_eval.json"
            }

        trial_config["data"].update(prepared_data[data_key])
        evaluation_dir = trial_dir / "evaluation"
        trial_config["evaluation"]["output_dir"] = str(evaluation_dir)
        trial_config_path = trial_dir / "config.yaml"

        for section in ("preprocessing", "model", "training"):
            for name, value in trial_config[section].items():
                if name not in ("output_dir", "resume_from"):
                    row[f"{section}_{name}"] = value

        print(f"Trial {index}/{len(combinations)}")

        with TemporaryDirectory(prefix="checkpoints_", dir=trial_dir) as folder:
            checkpoints = Path(folder)
            trial_config["training"]["output_dir"] = str(checkpoints)
            trial_config["evaluation"]["checkpoint_path"] = str(checkpoints / "best")
            save_config(trial_config, trial_config_path)

            if needs_preprocessing:
                run_script("preprocess.py", trial_config_path)

            run_script("train.py", trial_config_path)
            run_script("evaluate.py", trial_config_path)

            history_path = checkpoints / "last" / "training_history.json"
            best_history_path = checkpoints / "best" / "training_history.json"

            last_epoch = read_json(history_path)[-1]
            best_epoch = read_json(best_history_path)[-1]
            summary = read_json(evaluation_dir / "evaluation_summary.json")

            shutil.copy2(history_path, trial_dir / "training_history.json")
            plot_loss(history_path, trial_dir / "learning_curve.png")

            row.update(summary["overall"])
            row["best_epoch"] = best_epoch["epoch"]
            row["best_train_loss"] = best_epoch["train_loss"]
            row["best_validation_loss"] = best_epoch["validation_loss"]
            row["last_epoch"] = last_epoch["epoch"]
            row["last_train_loss"] = last_epoch["train_loss"]
            row["last_validation_loss"] = last_epoch["validation_loss"]

            score = (row["anls"], -row["best_validation_loss"])
            if best_score is None or score > best_score:
                if best_dir.exists():
                    shutil.rmtree(best_dir)

                shutil.move(str(checkpoints / "best"), str(best_dir))
                best_score = score

                best_config = copy.deepcopy(trial_config)
                best_config["training"]["output_dir"] = str(output_dir / "retrain")
                best_config["evaluation"]["checkpoint_path"] = str(best_dir)
                save_config(best_config, output_dir / "best_config.yaml")

                with (output_dir / "best.json").open("w", encoding="utf-8") as file:
                    json.dump(row, file, indent=2)

            rows.append(row)
            save_results(rows, output_dir / "results.csv")

    print("Results:", output_dir / "results.csv")
    print("Best model:", best_dir)

if __name__ == "__main__":
    main()