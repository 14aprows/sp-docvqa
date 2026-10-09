import csv
import json
import matplotlib.pyplot as plt

def plot_loss(history_path, output_path):
    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not history:
        raise ValueError(f"Training history is empty: {history_path}")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["validation_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Learning Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return min(history, key=lambda x: x["validation_loss"])

def save_results(rows, output_path):
    if not rows:
        raise ValueError("Grid search results are empty.")

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)