import pandas as pd

def test_fashion_metrics_10_classes():
    df = pd.read_csv("outputs/tables/fashion_test_results.csv")
    # 10 rows for classes
    assert len(df) == 10
    assert set(df.columns) == {"class", "precision", "recall", "f1", "support"}
    df2 = pd.read_csv("outputs/tables/fashion_test_results_agg.csv")
    assert set(df2['type']) == {"micro", "macro"}
