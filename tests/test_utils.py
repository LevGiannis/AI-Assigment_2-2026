import pytest
import numpy as np
import torch
from src.utils import seed, logger, device, metrics, io, plots
import pandas as pd
import os

def test_seed_determinism():
    seed.set_seed(123)
    a = np.random.rand(3)
    seed.set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)

def test_logger():
    log = logger.get_logger("test_logger")
    log.info("Logger works!")
    assert log.name == "test_logger"

def test_device():
    d = device.get_device()
    assert str(d) in ["cpu", "cuda", "mps"]

def test_metrics():
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 2, 1, 1, 0, 2]
    labels = [0, 1, 2]
    m = metrics.compute_metrics(y_true, y_pred, labels=labels)
    # Check shape
    assert len(m['per_class']['precision']) == 3
    assert 'micro' in m and 'macro' in m

def test_io(tmp_path):
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    f = tmp_path / "test.csv"
    io.save_table(df, str(f))
    df2 = io.load_table(str(f))
    assert df2.equals(df)

def test_plots(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1,2,3],[4,5,6])
    f = tmp_path / "plot.png"
    plots.save_plot(fig, str(f))
    assert os.path.exists(f)
