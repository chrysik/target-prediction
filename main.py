"""main."""
from rhiprediction.config import raw_data_folder, df_raw_types, df_target_types
from rhiprediction import DataPreprocessing
from rhiprediction import DataModeling
from pathlib import Path

if __name__ == '__main__':
    raw_df_path = Path(raw_data_folder) / 'md_raw_dataset.csv'
    target_df_path = Path(raw_data_folder) / 'md_target_dataset.csv'

    dp = DataPreprocessing(raw_path=raw_df_path,
                           target_path=target_df_path,
                           df_raw_types=df_raw_types,
                           df_target_types=df_target_types)

    df_preprocessed = dp.preprocess_dfs(auto_profiling=False,
                                        generate_plots=False)

    data_modeling = DataModeling(df_preprocessed)
    data_modeling.random_forest(random_search=False)
