import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="")


class DataModeling:
    """Creates the DataModeling class

     Construction:: dm = DataModeling(df="sample dataframe")

     Parameters
     ----------
     df: Dataframe
         Dataframe that contains the features and target values.
     """

    def __init__(self, df):
        self.df = df
        self.target = 'target'
        self.train_labels = np.array(self.df[self.target])
        self.train_features = np.array(self.df.drop([self.target], axis=1))

    def random_forest(self, random_search=False):

        """
        Trains and evaluates the random forest model.

        Parameters
        ----------
        random_search : bool
            Whether to perform randomized search for hyperparameter tuning

        Returns
        -------
        rf : Model
            Returns the trained random forest model

        """

        if random_search:
            best_params = self.__random_search()
            logging.info(f'\nBest parameters:\n {best_params}')
        else:
            best_params = {'n_estimators': 400,
                           'min_samples_split': 10,
                           'min_samples_leaf': 5,
                           'max_features': 'auto',
                           'max_depth': 20,
                           'bootstrap': True}

        rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                   min_samples_split=best_params['min_samples_split'],
                                   min_samples_leaf=best_params['min_samples_leaf'],
                                   max_features=best_params['max_features'],
                                   max_depth=best_params['max_depth'],
                                   bootstrap=best_params['bootstrap'],
                                   random_state=42)
        rf.fit(self.train_features, self.train_labels)
        self.__evaluate_model(rf)
        return rf

    def __random_search(self):
        """Performs randomized search for hyperparameter tuning"""
        n_estimators = [100, 200, 300]  # Number of trees
        max_features = ['auto', 'sqrt']  # Number of features to consider at every split
        max_depth = [10, 20]  # Maximum depth in tree
        min_samples_split = [5, 10]  # Minimum number of samples required to split a node
        min_samples_leaf = [3, 5]  # Minimum number of samples required at each leaf node
        bootstrap = [True, False]  # Method of selecting samples for training each tree

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
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=2, verbose=2,
                                       scoring='neg_mean_absolute_error')  # Fit the random search model
        rf_random.fit(self.train_features, self.train_labels)
        return rf_random.best_params_

    def __evaluate_model(self, model):
        """Evaluates the model

        Parameters
        ----------
        model : Model
            The trained and fitted model
        """
        predictions = model.predict(self.train_features)
        errors = abs(predictions - self.train_labels)
        logging.info(f'Mean Absolute Error: {round(np.mean(errors), 2)} degrees.')
