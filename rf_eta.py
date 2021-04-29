from sklearn.model_selection import train_test_split,cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import joblib


class RF(object):
    def __init__(self):
        self.name = 'rf'

        self.params = {
            'criterion': 'mae',
            'max_depth': 6,
            'n_estimators':120,
            'max_features': 0.9,
            'max_leaf_nodes': 24,
            'min_samples_split': 4,
            'min_samples_leaf': 10
        }

        self.grid_params = {
            'min_samples_split':[x for x in range(2,10)],
            'min_samples_leaf': [x for x in range(10,30,4)]
        }

        self.model = RandomForestRegressor(**self.params)

    def train_update(self,X,Y):
        grid = GridSearchCV(self.model,self.grid_params,cv=5,scoring='neg_mean_absolute_error',verbose=1,n_jobs=4)
        grid.fit(X,Y)
        print(grid.best_score_)
        print(grid.best_params_)

    def train(self,x,y):
        self.model.fit(x,y)

    def pred(self,x):
        pred = self.model.predict(x)
        return pred

if __name__ == '__main__':
    # rf = RF()
    # _train = pd.read_excel('./data/concat_RF.xlsx')
    # _train.drop(['new_label'],axis=1,inplace=True)
    #
    # pre_y = _train['label'].values
    # pre_x = _train.iloc[:,:-1].values
    #
    # train_x,test_x,train_y,test_y = train_test_split(pre_x,pre_y,test_size=0.3,random_state=12)
    #
    # rf.train(train_x,train_y)
    #
    # joblib.dump(rf,'./model/RF_eta.model')
    #
    # pred = rf.pred(test_x)
    #
    # total_loss = sum(abs(pred-test_y))/len(pred)
    # print(total_loss)


    rf = joblib.load('./model/RF_eta.model')

    _test = pd.read_excel('./data/eval/2021-2-4-labels.xlsx')
    _test.drop('new_label',axis=1,inplace=True)

    pre_y = _test['label'].values
    pre_x = _test.iloc[:,:-1].values

    pred = rf.pred(pre_x)

    # print(pred,pre_y)
    print(np.mean(pre_y))
    print(np.mean(pred))

    total_loss = 2*np.mean(np.abs(pred-pre_y)/np.add(np.abs(pred),np.abs(pre_y)))
    print(total_loss)
