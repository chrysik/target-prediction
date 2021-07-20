"""DataPreprocessing."""
import pandas as pd
from pandas_profiling import ProfileReport
import logging
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import math
import seaborn as sns
import os
import warnings

pd.set_option('display.max_columns', 40)
warnings.filterwarnings(action='ignore')
logging.basicConfig(level=logging.INFO, format="")


class DataPreprocessing:
    """Create the DataPreprocessing class.

    Construction:: dp = DataPreprocessing(raw_path="sample Raw df path",
    target_path="sample Target df path",df_raw_types="sample Raw data types",
    df_target_types="sample Target data types")

    Parameters
    ----------
    raw_path: str
        Raw dataset path
    target_path: str
        Target dataset path
    df_raw_types: dict, None
        Raw dataset data types
    df_target_types: dict, None
        Target dataset data types
    """

    def __init__(self, raw_path=None, target_path=None,
                 df_raw_types=None, df_target_types=None):
        """Initialize DataPreprocessing class."""
        self.df_raw_path = raw_path
        self.df_target_path = target_path
        self.df_raw_types = df_raw_types
        self.df_target_types = df_target_types
        self.df_raw_name = 'Raw'
        self.df_target_name = 'Target'
        self.target = 'target'

    def __data_loader(self):
        """Load the datasets and create a dictionary of them."""
        df_raw = pd.read_csv(self.df_raw_path, delimiter=";")
        df_target = pd.read_csv(self.df_target_path, delimiter=";")
        self.__dfs_to_dictionary(df_raw, df_target)

    def preprocess_dfs(self, auto_profiling=False, generate_plots=False):
        """Preprocess the datasets Raw and Target.

        Preprocess datasets so that they are ready to by consumed by the model.
        Includes EDA, cleaning, feature engineering, feature selection
        and data transformation.

        Parameters
        ----------
        auto_profiling : bool
            Whether to execute the profiling using pandas-profiling
        generate_plots : bool
            Whether to generate exploratory analysis plots

        Returns
        -------
        df : DataFrame
            Returns the merged and clean dataframe
        """
        if not isinstance(auto_profiling, bool):
            raise AttributeError("auto_profiling must be True or False")
        if not isinstance(generate_plots, bool):
            raise AttributeError("generate_plots must be True or False")
        self.auto_profiling = auto_profiling
        self.generate_plots = generate_plots

        self.__data_loader()
        self.__clean_column_names()
        self.__initial_profiling()
        self.__fix_dtypes()
        self.__drop_duplicate_obs()
        self.__initial_eda()
        self.__find_index_column()
        self.__find_dataset_key()
        self.__merge_dfs()
        self.__merged_df_eda()
        # Here handle outliers
        self.__feature_engineering()
        self.__handle_nan_values()
        self.__feature_selection()  # Add an anova and a backpropagation
        self.__data_transformation()
        return self.df_merged

    def __dfs_to_dictionary(self, df_raw, df_target):
        """Add Raw and Target dataset in a dictionary.

         Contain information such as name and datatypes.

        Parameters
        ----------
        df_raw : DataFrame
            Raw dataframe
        df_target : DataFrame
            Target dataframe
        """
        self.dfs = {self.df_raw_name: {'df': df_raw,
                                       'df_dtypes': self.df_raw_types},
                    self.df_target_name: {'df': df_target,
                                          'df_dtypes': self.df_target_types}
                    }

    def __clean_column_names(self):
        """Clean the column names."""
        logging.info(f'Cleaning columns...')
        for name, df_dict in self.dfs.items():
            for col in df_dict['df'].columns:
                df_dict['df'].rename(
                    columns={col: col.lower().strip().replace(":", "")
                        .replace(" ", "_")}, inplace=True)

    def __initial_profiling(self):
        """Execute a profiling on the initial data."""
        if self.auto_profiling:
            logging.info(f'Initiating data profiling...')
            if not os.path.exists("profiling"):
                os.makedirs("profiling")
            for name, df_dict in self.dfs.items():
                prof = ProfileReport(df_dict['df'], minimal=True)
                prof.to_file(output_file=f'profiling/{name}_Profiling.html')
        else:
            pass

    def __fix_dtypes(self):
        """Change datatypes on the data, according to the information given."""
        logging.info(f'Fixing data types...')
        for name, df_dict in self.dfs.items():
            if df_dict['df_dtypes']:
                # Datetime variables
                for i in df_dict['df_dtypes']['datetime_cols']:
                    df_dict['df'][i] = pd.to_datetime(df_dict['df'][i],
                                                      errors='coerce')
                # Integer variables
                for i in df_dict['df_dtypes']['int_cols']:
                    df_dict['df'][i] = pd.to_numeric(df_dict['df'][i],
                                                     errors='coerce',
                                                     downcast='integer')
                # Float variables
                for i in df_dict['df_dtypes']['float_cols']:
                    df_dict['df'][i] = pd.to_numeric(df_dict['df'][i],
                                                     errors='coerce')
                # Categorical variables
                for i in df_dict['df_dtypes']['categ_cols']:
                    df_dict['df'][i] = df_dict['df'][i].astype('category')

            logging.info(
                f'\nDatatypes for {name} dataset: \n{df_dict["df"].dtypes}\n')

    def __drop_duplicate_obs(self):
        """Drop duplicate observations."""
        logging.info(f'\nDropping duplicate rows...')
        for name, df_dict in self.dfs.items():
            len_init = df_dict['df'].shape[0]
            df_dict['df'] = df_dict['df'].drop_duplicates()
            logging.info(
                f'Dropped {len_init - df_dict["df"].shape[0]}'
                f'rows from the {name} dataset.')

    def __initial_eda(self):
        """Perform initial exploratory analysis, generating Histograms."""
        for name, df_dict in self.dfs.items():
            logging.info(
                f'\nDescriptive Statistics for {name} dataset:\n'
                f'{df_dict["df"].describe()}')

        if self.generate_plots:
            if not os.path.exists("plots"):
                os.makedirs("plots")
            for name, df_dict in self.dfs.items():
                # For categorical variables
                len_cat = len(df_dict['df'].select_dtypes(
                    include=['category', 'object']).columns)
                if len_cat > 0:
                    fig, ax = plt.subplots(4, math.ceil(len_cat / 4),
                                           figsize=(len_cat * 2, len_cat * 3))
                    for variable, subplot in zip(
                            df_dict['df'].select_dtypes(
                                include=['category', 'object'])
                                    .columns, ax.flatten()):
                        sns.countplot(df_dict['df'][variable],
                                      ax=subplot).set_title(
                            f'Histogram of {variable}')
                        for label in subplot.get_xticklabels():
                            label.set_rotation(90)
                    ax.flat[-1].set_visible(False)
                    plt.savefig(f'plots/{name}_Histograms_Categorical.png')

                # For numerical variables
                numer_cols = df_dict['df'].select_dtypes(
                    exclude=['category', 'object', 'datetime']).columns
                if len(numer_cols) > 0:
                    df_dict['df'][numer_cols].hist(bins=15, figsize=(
                        len(numer_cols) * 2, len(numer_cols) * 3),
                                                   layout=(math.ceil(
                                                       len(numer_cols) / 4),
                                                           4),
                                                   label='Count')
                    plt.savefig(f'plots/{name}_Histograms_Numerical.png',
                                bbox_inches='tight')

    def __find_index_column(self):
        """Find the index column in Raw dataset."""
        # Check if "index" in Raw dataset
        if 'index' in self.dfs['Raw']['df'].columns:
            pass
        else:
            logging.info("\nIndex columns is missing from the Raw dataset.")
            similarity_dict = {}
            # To find potential index column in Raw dataset, compare all Raw
            # columns with Index column in Target dataset
            for un_col in self.dfs['Raw']['df'].columns:
                res = len(
                    set(self.dfs['Raw']['df'][un_col].unique()) & set(
                        self.dfs['Target']['df']["index"].unique())) / \
                      float(len(
                          set(self.dfs['Raw']['df'][un_col].unique()) | set(
                              self.dfs['Target']['df'][
                                  "index"].unique()))) * 100
                similarity_dict[un_col] = res
            max_sim_col = max(similarity_dict, key=similarity_dict.get)
            logging.info(
                f'Column "{max_sim_col}" has similarity percentage with '
                f'Target Index: {similarity_dict[max_sim_col]}%')
            # Rename it to index
            self.dfs['Raw']['df'] = self.dfs['Raw']['df'].rename(
                columns={max_sim_col: "index"})

    def __find_dataset_key(self):
        """Find the key of the datasets."""
        # For both the Target and Raw datasets, the colums "index" and "groups"
        # consist the key of the tables
        for name, df_dict in self.dfs.items():
            if df_dict['df'][
                df_dict['df'].duplicated(subset=['index', 'groups'],
                                         keep=False)].shape[0] == 0:
                logging.info(f'They key of the dataset {name} '
                             f'is the combination "Index, Group"')
            self.dfs[name]['df'] = df_dict['df'].drop_duplicates(
                subset=['index', 'groups'], keep=False)

    def __merge_dfs(self):
        """Merge Raw and Target dataframes."""
        logging.info(f'\nMerging datasets...')
        self.df_merged = pd.merge(self.dfs['Raw']['df'],
                                  self.dfs['Target']['df'], how='inner',
                                  left_on=['index', 'groups'],
                                  right_on=['index', 'groups'])
        logging.info(f'Size of the merged dataset: {self.df_merged.shape}')

        self.df_merged = self.df_merged.set_index(['index', 'groups'])

    def __merged_df_eda(self):
        """Perform analysis on the merged dataset.

        Generate Boxplots, Scatterplots, Correlation and Autocorrelation plots
        """
        if self.generate_plots:
            # Remove unused categories
            categ_col = self.df_merged.select_dtypes(
                include=['category', 'object']).columns
            for col in categ_col:
                self.df_merged[col] = self.df_merged[
                    col].cat.remove_unused_categories()

            # Boxplots with target
            fig, ax = plt.subplots(4, math.ceil(len(categ_col) / 4), figsize=(
                len(categ_col) * 2, len(categ_col) * 3))
            for variable, subplot in zip(categ_col, ax.flatten()):
                sns.boxplot(x=variable, y=self.target, data=self.df_merged,
                            ax=subplot)
            plt.savefig(f'plots/Merged_dataset_boxplots_with_target.png',
                        bbox_inches='tight')
            logging.info('\nMerged dataset boxplots exported.')

            # Scatterplots with target
            numer_cols = self.df_merged.select_dtypes(
                exclude=['category', 'object', 'datetime']).columns
            fig, ax = plt.subplots(3, 3, figsize=(15, 10))
            for variable, subplot in zip(numer_cols, ax.flatten()):
                sns.scatterplot(x=variable, y=self.target, data=self.df_merged,
                                ax=subplot)
            plt.savefig(f'plots/Merged_dataset_scatterplots_with_target.png',
                        bbox_inches='tight')
            logging.info('Merged dataset scatterplots exported.')

            # Correlation plot
            fig, ax = plt.subplots(figsize=(17, 12))
            sns.heatmap(self.df_merged.corr(method='pearson'), annot=True,
                        fmt='.4f',
                        cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
            plt.savefig(f'plots/Merged_dataset_correlations.png')
            logging.info('Correlation plot exported.')

            # Autocorrelation plot
            tsaplots.plot_acf(
                self.df_merged.sort_values(by=['when'])['target'], lags=40)
            plt.savefig(f'plots/Autocorrelation_target.png')
            logging.info('Autocorrelation plot exported.')

            # Plot numerical variables using timestamp, for the first index
            merged_oneindex = self.df_merged.reset_index()
            merged_oneindex = merged_oneindex[merged_oneindex['index'] == 1]
            _, ax = plt.subplots(3, 3, figsize=(15, 10))
            for variable, subplot in zip(numer_cols, ax.flatten()):
                sns.lineplot(x='start_process', y=variable,
                             data=merged_oneindex, ax=subplot)
            plt.savefig(
                f'plots/Merged_dataset_scatterplot_with_timestamp_index_1.png',
                bbox_inches='tight')

            tsaplots.plot_acf(
                merged_oneindex.sort_values(by=['when'])['target'], lags=40)
            plt.savefig(f'plots/Autocorrelation_in_target_index_1.png')
            logging.info(
                'Merged dataset scatterplots with_timestamp exported index 1.')

    def __feature_engineering(self):
        """Perform feature engineering.

        Create new columns from the datetime columns.
        """

        def timestamp_substractions(df_merged, newcol, col1, col2):
            df_merged[newcol] = df_merged[col2] - df_merged[col1]
            df_merged[newcol] = df_merged[newcol].astype('timedelta64[m]')
            return df_merged

        # Use timestamp columns by creating new ones that could be useful
        self.df_merged = timestamp_substractions(
            self.df_merged, 'process_duration', 'start_process', 'process_end')
        self.df_merged = timestamp_substractions(
            self.df_merged, 'subprocess1_duration', 'start_subprocess1',
            'subprocess1_end')
        self.df_merged = timestamp_substractions(
            self.df_merged, 'time_to_start_subprocess1', 'start_process',
            'start_subprocess1')
        self.df_merged = timestamp_substractions(
            self.df_merged, 'time_to_start_critsubprocess1',
            'start_subprocess1', 'start_critical_subprocess1')
        # Remove datetime cols:
        self.df_merged = self.df_merged.drop(
            self.df_merged.select_dtypes(include=['datetime']).columns, axis=1)

    def __handle_nan_values(self):
        """Handle nan values."""
        # Log percent missing
        percent_missing = self.df_merged.isnull().sum() * 100 / len(
            self.df_merged)
        missing_value_df = pd.DataFrame(
            {'column_name': self.df_merged.columns,
             'percent_missing': percent_missing.round(1)})
        logging.info(f'\n Mining values:\n {missing_value_df}')

        # Delete the categorical missing
        for col in self.df_merged.select_dtypes(
                include=['category', 'object']).columns:
            self.df_merged = self.df_merged[self.df_merged[col].notna()]
        # Nan to zeros for numerical variables
        for col in self.df_merged.select_dtypes(
                include=['int16', 'int32', 'float']).columns:
            self.df_merged[col] = self.df_merged[col].fillna(0)
        logging.info(
            f'{self.df_merged[self.df_merged.isna().any(axis=1)].shape[0]}'
            f'observations with NaN values left.')
        logging.info('NaN values handled.')

    def __feature_selection(self):
        """Perform feature selection."""

        def remove_highly_cor_values(df, cor=0.8):
            logging.info('\nRemove highly correlated variables.')
            correlation_matrix = df.corr()
            correlated_features = set()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) >= cor:
                        logging.info(
                            f'Variables{correlation_matrix.columns[i]}'
                            f' and {correlation_matrix.columns[j]} are'
                            f' correlated with correlation coefficient'
                            f'{abs(correlation_matrix.iloc[i, j]).round(2)}')
                        colname = correlation_matrix.columns[i]
                        correlated_features.add(colname)
            # Drop them
            df = df.drop(list(correlated_features), axis=1)
            logging.info(f'Variables {list(correlated_features)}'
                         f'have been removed.')
            return df

        self.df_merged = remove_highly_cor_values(self.df_merged, cor=0.8)

    def __data_transformation(self):
        """Perform data transformation.

        Convert categorical variable into dummy/indicator variables.
        Aggregating rows with equal feature values and different target values.
        """
        # Categorical variables to dummy
        logging.info('\nTransforming categorical variables to dummy...')
        self.df_merged = pd.get_dummies(self.df_merged, drop_first=True)
        # Aggregation
        logging.info('\nAggregating observations with equal'
                     'feature values and different target values...')
        logging.info('Examples of observations with same values'
                     'in all features except from the target value:\n')
        example_agg = self.df_merged[self.df_merged.duplicated(
            subset=[i for i in self.df_merged.columns if i != self.target],
            keep=False)].sort_values("crystal_weight", ascending=False).head(6)
        logging.info(f'{example_agg}')
        self.df_merged = self.df_merged \
            .groupby([i for i in self.df_merged.columns if i != self.target],
                     as_index=False).agg({'target': 'mean'})
