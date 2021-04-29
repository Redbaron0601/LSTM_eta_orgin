from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import lightgbm as lgb
import pandas as pd
import numpy as np
# import gc
import os
# import json
# import pickle
import CONSTANT
from sklearn.model_selection import ShuffleSplit, train_test_split, TimeSeriesSplit, KFold
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import make_scorer
# from datetime import date, timedelta

pd.set_option('display.max_rows', 100)


class LGBModel:
    def __init__(self):

        self.name = 'lgb'
        self.type = 'tree'

        self.params = {
            "boosting_type": "gbdt",
            "objective": 'mape',
            "seed": '2020',
            "num_threads": 4,
            "verbose": 1,
            "metric": ''
        }

        self.hyperparams = {
            'num_leaves': 51,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0002,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'learning_rate': 0.03,
            'num_iterations':500
        }

        self.model = None

    def train(self, X_train, y_train, cat):
        def my_object(preds, train_data):
            labels = train_data.get_label()
            diff = preds - labels
            index = preds >= labels
            grad = diff
            grad[index] *= CONSTANT.W
            hess = np.where(diff > 0, 2, 2.0)
            return grad, hess
        # While training,use bayes_opt to tune params
        # self.bayes_opt(X_train, y_train, cat)
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train({**self.params, **self.hyperparams}, num_boost_round=500, fobj=my_object,
                               train_set=lgb_train, valid_sets=[lgb_train], categorical_feature=cat)

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return preds

    def save(self, path, mode):
        file_path = os.path.join(path, f'{mode}_lgb.txt')
        self.model.save_model(filename=file_path)

    def load(self, path, MODE):
        file_path = os.path.join(path, f'{MODE}_lgb.txt')
        self.model = lgb.Booster(model_file=file_path)

    def bayes_opt(self, X, y, cat):
        X_opt, X_eval, y_opt, y_eval = train_test_split(X, y, test_size=0.2, shuffle=True)
        train_data = lgb.Dataset(X_opt, label=y_opt)
        valid_data = lgb.Dataset(X_eval, label=y_eval)

        params = self.params.copy()

        space = {
            'max_depth': hp.choice("max_depth", [-1, 3, 5, 7, 9, 11]),
            "num_leaves": hp.choice("num_leaves", np.arange(20, 128, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "min_child_samples": hp.choice("min_data_in_leaf", np.arange(10, 120, 10, dtype=int)),
            "min_child_weight": hp.uniform('min_child_weight', 0.01, 1),
            "min_split_gain": hp.uniform('min_split_gain', 0.001, 0.1),
            'colsample_bytree': hp.choice("colsample_bytree", [0.7, 0.9]),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
        }

        def objective(hyperparams):
            print('hyperparams {}'.format(hyperparams))
            model = lgb.train({**params, **hyperparams},num_boost_round=100,
                              train_set=train_data, valid_sets=valid_data,
                              early_stopping_rounds=20,verbose_eval=False)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=50, verbose=1,
                             rstate=np.random.RandomState(1))
        self.hyperparams.update(space_eval(space, best))
        print("auc = {}, hyperparams: {}".format(-trials.best_trial['result']['loss'], self.hyperparams))
        self.early_stop_opt(X_opt, X_eval, y_opt, y_eval, cat)
        return

    def early_stop_opt(self, X_opt, X_eval, y_opt, y_eval, cat):
        lgb_train = lgb.Dataset(X_opt, y_opt)
        lgb_eval = lgb.Dataset(X_eval, y_eval)
        model = lgb.train({**self.params, **self.hyperparams}, num_boost_round=self.hyperparams['num_iterations'], verbose_eval=20,
                          train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval',
                          categorical_feature=cat,
                          early_stopping_rounds=20)
        self.hyperparams['num_iterations'] = model.best_iteration
        return

    def log_feat_importances(self, return_info=False):
        importances = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                                    'importances': self.model.feature_importance("gain")})# also could use gini

        importances.sort_values('importances', ascending=False, inplace=True)

        print('feat importance:{}'.format(importances.head(100)))
        # Considering using matplotlib to visualize the feature importance
