import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import plot_importance

# xgb.__version__ : 1.4.2

def train_test_split(df,label,begin_index, train_length, test_length,lags):
    """
    split training set and test set for time series prediction
    df: dataset
    label: name of the label column
    begin_index: where the training set starts
    train_length: number of observations of the training set
    test_length: number of observations of the test set
    lags: lags between the end of training set and the start of test set, to avoid data snooping bias
    """
    train_end = begin_index + train_length
    test_end = train_end + test_length + lags
    train = df.iloc[begin_index:train_end,:]
    test = df.iloc[train_end+lags:test_end,:]
    X_train = train.drop(labels=[label],axis=1)
    X_test = test.drop(labels=[label], axis=1)
    y_train = train[label]
    y_test = test[label]
    return X_train,X_test,y_train, y_test

def train_xgb(dt,label,begin_index, train_length, test_length,lags,is_silent,objective,subsample,max_depth,
              colsample_bytree,learning_rate,nfold=5,num_class=0,num_boost_round=50, early_stopping_rounds=10,plot=True):
    """
    split training set and test set for time series prediction
    dt: pandas dataframe, dataset
    label: string, name of the label column
    begin_index: int, where the training set starts
    train_length: int, number of observations of the training set
    test_length: int, number of observations of the test set
    lags: int, lags between the end of training set and the start of test set, to avoid data snooping bias
    is_silent: bool, controls results printing
    objective: string, objective function, 'multi:softprob' or 'reg-linear' in this case
    subsample: float, random sampling proportion when training a tree
    max_depth: int, length of a tree
    colsample_bytree: float, the subsample ratio of columns when constructing each tree.
    learning_rate: float, controls model complexity
    nfold: int, number of folds in cross validation
    num_class: int, number of classes in the label variables when performing multi-class classification
    num_boost_round: int,
    early_stopping_rounds: int, activates early stopping
    plot: bool, boolean indicator of whether show plots of cross validation results and feature importance
    """

    try:
        dt1 = dt.drop(columns=["date", "trade_date", "ts_code"])
    except:
        pass

    if objective == 'multi:softprob':
        dt1[label] = dt1[label].astype('int', errors='ignore') + 1  # label must be in [0, num_class)
    else:
        dt1[label] = dt1[label].shift(-1) # the regressand should be the next day's

    X_train, X_test, y_train, y_test = train_test_split(dt1, label, begin_index, train_length,test_length,lags)

    # cross validation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {'silent': is_silent,
             'objective': objective,
             "subsample": subsample,
             "max_depth": max_depth,
             "colsample_bytree": colsample_bytree,
             "learning_rate": learning_rate,
             "nfold": nfold,
             'shuffle': False}

    if objective == 'multi:softprob': param['num_class'] = num_class
    else: param['metrics'] = 'rmse'
    cvresult = xgb.cv(params=param, dtrain=dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds) # for classifier, the default metrics is mlogloss

    #best_num_round = np.argmin(cvresult['test-mlogloss-mean'])+1
    best_num_round = np.argmin(cvresult.iloc[:,2])+1

    if plot:
        fig, ax = plt.subplots(1, figsize=(16, 9))
        ax.grid()
        ax.plot(range(1,cvresult.shape[0]+1), cvresult.iloc[:, 0], c="red", label="train")
        ax.plot(range(1,cvresult.shape[0]+1), cvresult.iloc[:, 2], c="blue", label="validation")
        ax.legend(fontsize="xx-large")
        plt.axvline(best_num_round,linestyle='--', label=str(best_num_round))
        plt.legend()
        plt.show()


    #best_num_round = np.argmin(cvresult['test-mlogloss-mean'])+1
    if objective == 'multi:softprob':
        model = XGBClassifier(param, dtrain,n_boost_round=best_num_round)
        #model = xgb.train(params=param,dtrain=dtrain,num_boost_round=best_num_round,obj=objective)
        model.fit(X_train, y_train)
        y_fit = model.predict(X_train)
        in_acc = accuracy_score(y_train, y_fit)  # in of sample accuracy

        y_pred = model.predict(X_test)
        oos_acc = accuracy_score(y_test, y_pred)  # out of sample accuracy

        print('in sample accuracy：%.2f%%' % (in_acc * 100))
        print('out of sample accuracy：%.2f%%' % (oos_acc * 100))

        if plot:plot_importance(model)

        return y_pred,in_acc,oos_acc,model,cvresult

    else:
        model = XGBRegressor(objective=objective, silent= is_silent,subsample=subsample,
                             max_depth=max_depth,colsample_bytree=colsample_bytree,learning_rate=learning_rate,
                             nfold=nfold,shuffle=False,n_estimators=best_num_round)
        model.fit(X_train, y_train)
        y_fit = model.predict(X_train)
        in_mae = mean_absolute_error(y_train, y_fit)  # in sample mean absolute error
        in_mse = mean_squared_error(y_train, y_fit)  # in sample mean squared error

        y_pred = model.predict(X_test)
        try:
            oos_mae = mean_absolute_error(y_test, y_pred)  # out of sample mean absolute error
            oos_mse = mean_squared_error(y_test, y_pred)  # out of sample mean squared error

            print('in sample mean absolute error：%.2f' % (in_mae))
            print('in sample mean squared error：%.2f' % (in_mse))

            print('out of sample mean absolute error：%.2f' % (oos_mae))
            print('out of sample mean squared error：%.2f' % (oos_mse))
        except:
            oos_mae = 0
            oos_mse = 0

        if plot: plot_importance(model)

        return y_pred,in_mae,in_mse,oos_mae,oos_mse,model,cvresult


if __name__=="__main__":
    """
    df = pd.read_csv("data\\SH000001@0825_lags.csv", index_col=0, parse_dates=['date'])
    #df.drop(columns=["date","trade_date","ts_code"],inplace=True)
    y_pred,in_acc,oos_acc,model,cvresult = train_xgb(df,"label",61,100,20,5,True,'multi:softprob',1,6,0.5,0.1,5,3,500,10,True)
    y_pred2, in_mae,in_mse,oos_mae,oos_mse, model2, cvresult2 = train_xgb(df, "close", 61, 100, 20, 1,True, 'reg:linear',1,6,0.5,0.1,5,3,500,10,True)

    plt.scatter(df["date"][62:182],df["close"][62:182],cmap = 'brg', s = 30, alpha = 0.3, marker = '8', linewidth = 0,label='true')
    plt.scatter(df["date"][162:182],y_pred2,cmap = 'brg', s = 30, alpha = 0.3, marker = '8', linewidth = 0,label='predicted')
    plt.xlabel("date")
    plt.xlim(xmin=df["date"][62],xmax=df["date"][182])
    plt.ylim(ymin=2000,ymax=5000)
    plt.ylabel("closing price")
    plt.legend()
    plt.show()
    """
