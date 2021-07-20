from rhiprediction import modeling
import pytest
import pandas as pd

df = pd.read_csv("tests/sample_preprocessed_dataframe.csv")
dm = modeling.DataModeling(df)


@pytest.mark.parametrize(
    ("random_search"),
    [
        (True),
        (False),
        (None)
    ],
)
def test_random_forest(random_search):
    dm.random_forest(random_search=random_search)