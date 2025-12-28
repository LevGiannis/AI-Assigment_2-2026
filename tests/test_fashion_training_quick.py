import os
import sys
import subprocess
import pytest

def test_fashion_training_quick(tmp_path):
    config_path = tmp_path / "fashion_cnn.json"
    with open(config_path, "w") as f:
        f.write('{"batch_size": 4, "lr": 0.01, "epochs": 2, "seed": 42}\n')
    # Train quick
    subprocess.run([
        sys.executable, "-m", "src.fashion.train", "--config", str(config_path), "--quick"
    ], check=True)
    # Check artifacts
    assert os.path.exists("outputs/checkpoints/fashion_best.pt")
    assert os.path.exists("outputs/plots/fashion_loss_curves.png")
    # Eval quick
    subprocess.run([
        sys.executable, "-m", "src.fashion.eval_test", "--config", str(config_path), "--quick"
    ], check=True)
    assert os.path.exists("outputs/tables/fashion_test_results.csv")
