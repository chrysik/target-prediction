"""test preprocessing."""
from rhiprediction import DataPreprocessing
import pytest

dp = DataPreprocessing(raw_path='raw_data/md_raw_dataset.csv',
                       target_path='raw_data/md_target_dataset.csv')


@pytest.mark.parametrize(
    ("auto_profiling", "generate_plots"),
    [
        (None, False)
    ],
)
def test_preprocess_dfs(auto_profiling, generate_plots):  # noqa: D103
    dp.preprocess_dfs(auto_profiling=auto_profiling,
                      generate_plots=generate_plots)
