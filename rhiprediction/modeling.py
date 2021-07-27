"""DataModeling."""
import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="",
                    filename='logs.txt', filemode='a')


class DataModeling:
    """Create the DataModeling class.

    Construction:: dm = DataModeling(df="sample dataframe")

    Parameters
    ----------
    df: Dataframe
        Dataframe that contains the features and target values.
    """

    def __init__(self, df):
        """Initialize DataModeling class."""
        self.df = df
        self.target = 'target'
        self.all_labels = np.array(self.df[self.target])
        self.all_features = np.array(self.df.drop([self.target], axis=1))
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_labels = np.array(train[self.target])
        self.train_features = np.array(train.drop([self.target], axis=1))
        self.test_labels = np.array(test[self.target])
        self.test_features = np.array(test.drop([self.target], axis=1))

    def random_forest(self, random_search=False):
        """Train and evaluate the random forest model.

        Parameters
        ----------
        random_search : bool
            Whether to perform randomized search for hyperparameter tuning

        Returns
        -------
        rf : Model
            Returns the trained random forest model
        """
        if not isinstance(random_search, bool):
            raise AttributeError("random_search must be True or False")
        if random_search:
            best_params = self.__random_search()
            logging.info(f'\nBest parameters:\n {best_params}')
        else:
            best_params = {'n_estimators': 400,
                           'min_samples_split': 10,
                           'min_samples_leaf': 10,
                           'max_features': 'auto',
                           'max_depth': 15,
                           'bootstrap': True}

        rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            max_depth=best_params['max_depth'],
            bootstrap=best_params['bootstrap'],
            random_state=42)

        logging.info(f'\nTrain set size: {self.train_features.shape[0]} rows.'
                     f'\nTest set size: {self.test_features.shape[0]} rows.\n')

        rf.fit(self.train_features, self.train_labels)

        self.__evaluate_model(rf)
        return rf

    def __random_search(self):
        """Perform randomized search for hyperparameter tuning."""
        n_estimators = [100, 200, 300]
        max_features = ['auto', 'sqrt']
        max_depth = [10, 20]
        min_samples_split = [5, 10]
        min_samples_leaf = [3, 5]
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        logging.info(f'\nHyperparameter tuning using {random_grid}')

        # Use the random grid to search for best hyperparameters
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(
            estimator=rf, param_distributions=random_grid,
            n_iter=100, cv=2, verbose=2,
            scoring='neg_mean_absolute_error')
        rf_random.fit(self.all_features, self.all_labels)
        return rf_random.best_params_

    def __evaluate_model(self, model):
        """Evaluate the model.

        Parameters
        ----------
        model : Model
            The trained and fitted model
        """

        def plot_expected_predicted(setname, features, labels, model, errors):
            predictions = model.predict(features)
            # print('MAE: %.3f' % mae)
            # Plot expected vs predicted
            fig, ax = plt.subplots(figsize=(16, 6))
            plt.plot(labels, label='True values', color='red')
            plt.plot(predictions, 'ro', label='Predicted values', color='blue')
            plt.title(f'Expected vs Predicted values on {setname} set')
            plt.ylabel("Target")
            plt.xlabel(f'{setname} set Observations')
            plt.xticks([])
            ax.text(0.95, 0.01, f'Mean Absolute Error: {errors}', fontsize=10,
                    bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
            plt.legend()
            plt.savefig(f'plots/Prediction_vs_{setname}_set_results.png')

            _, _ = plt.subplots(figsize=(16, 6))
            plt.plot(labels - predictions, marker='o', linestyle='')
            plt.xticks([])
            plt.title(f'Prediction Errors on {setname} set')
            plt.ylabel("Error")
            plt.xlabel(f'{setname} set Observations')
            plt.savefig(f'plots/Prediction_Errors_{setname}_set_results.png')

            logging.info(f'Min prediction value:'
                         f'{predictions.min().round(2)}\n'
                         f'Max prediction value:'
                         f'{predictions.max().round(2)}\n'
                         f'Mean prediction values:'
                         f'{predictions.mean().round(2)}')
            logging.info(f'Min true value: {labels.min().round(2)}\n'
                         f'Max true value: {labels.max().round(2)}\n'
                         f'Mean true values: {labels.mean().round(2)}\n')

        predictions_train = model.predict(self.train_features)
        errors_train = abs(predictions_train - self.train_labels)
        predictions_test = model.predict(self.test_features)
        errors_test = abs(predictions_test - self.test_labels)
        mean_abs_error_train = round(np.mean(errors_train), 2)
        logging.info(f'Mean Absolute Error in Train set: '
                     f'{mean_abs_error_train} degrees.')
        mean_abs_error_test = round(np.mean(errors_test), 2)
        logging.info(f'Mean Absolute Error in Test set: '
                     f'{mean_abs_error_test} degrees.')

        # Visualize predicted vs expected values over time
        plot_expected_predicted('Test', self.test_features,
                                self.test_labels, model, mean_abs_error_test)
        plot_expected_predicted('Train', self.train_features,
                                self.train_labels, model, mean_abs_error_train)
