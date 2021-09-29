import pandas as pd
import feature_engineering as fe
import XGB
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import time


def create_window(dt,begin_ind,end_ind,window_length,method,forward = 0):
    """
    create window when performing predictions
    dt:
    begin_ind: int, begin index of all windows
    end_ind: int, end index of all windows
    window_length: int, length of the window
    method: string, "fixed one": 1 fixed window, "rolling": rolling window, "recursive": fixed start with an expanding window length
    forward: int, number of steps that the end of the window moves forward
    """
    # window_list = []
    #dt_w = dt.iloc[begin_ind:end_ind,:]
    begin_w, end_w = begin_ind, begin_ind + window_length
    dt_w = dt.iloc[begin_w:end_w,:]
    if method == "fixed one":
        if end_w <= end_ind:
            yield dt_w
        else:
            print("`window_length` parameter should not exceed the length of `dt`")

    if method == "rolling":
        while end_w <= end_ind:
            dt_w = dt.iloc[begin_w:end_w, :]
            begin_w += forward
            end_w += forward
            yield dt_w

    if method == "recursive":
        while end_w <= end_ind:
            dt_w = dt.iloc[begin_w:end_w, :]
            end_w += forward
            yield dt_w


def plot_result(dt,train_begin_ind,test_begin_ind,test_end_ind,y_pred,type,file_name):
    """
        plot predictions on the test set.
        begin_ind: int, beginning of the test set, if the prediction is the next day's closing price, the index should be of the next day's
        end_ind: int, end of the test set
        label: string, the target variable, either 'label' or 'close'
        y_pred: list or array, predicted value, its length should match 'begin_ind' and 'end_ind', note that its length should match `begin_ind` and `end_ind`
        file_name: string, file name of the saved figure
        """
    df = dt.iloc[train_begin_ind:test_end_ind,:]
    df1 = dt.iloc[test_begin_ind:test_end_ind,:] # classification
    df1["pred"] = y_pred

    df_ = dt.iloc[train_begin_ind+1:test_end_ind+1, :]
    df2 = dt.iloc[test_begin_ind+1:test_end_ind+1,:] # regression, the target variables is the next day's closing price
    df2["pred"] = y_pred
    plt.style.use('ggplot')
    fig1 = plt.figure(figsize=(16, 9))
    if type=='label':
        colors = ['g', 'r', 'b']
        Label_Com = ['predicted up', 'predicted down', 'predicted oscillate']
        label_list = [2, 0, 1]
        for item in label_list:
                Price = df1.loc[df1['pred'] == item]['close']
                Index = df1.loc[df1['pred'] == item]['date']
                index = label_list.index(item)
                plt.scatter(Index, Price, c=colors[index], cmap='brg', s=20, alpha=0.3, marker='8', linewidth=0,label='predicted')
        ax = fig1.gca()
        for lab in ax.xaxis.get_ticklabels():
            lab.set_rotation(30)

        # added this to get the legend to work
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels=Label_Com, loc='upper right')
        plt.ylim(ymin=2000, ymax=4000)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')

        ax2 = ax.twinx()
        ax2.plot(df.date, df.close, c='b', alpha=0.2,label='realized')

        #plt.xlim(xmin=df["date"][begin_ind], xmax=df["date"][end_ind])
        plt.ylim(ymin=2000, ymax=4000)
        plt.savefig("out\\figs\\{}predictLabel.jpg".format(file_name))
        plt.show()


    if type == "close":
        plt.style.use('ggplot')
        fig2 = plt.figure(figsize=(16, 9))
        plt.plot(df_["date"], df_["close"], c='b', alpha=0.2, label='realized')
        plt.scatter(df2["date"], df2["pred"], c='b', s=20, alpha=0.3, marker='8', linewidth=0, label='predicted')
        plt.xlabel("Date")
        # plt.xlim(xmin=df["date"][begin_ind], xmax=df["date"][end_ind])
        plt.ylim(ymin=2000, ymax=4000)
        plt.ylabel("Closing price")
        plt.legend()
        plt.savefig("out\\figs\\{}predictClose.jpg".format(file_name))
        plt.show()



if __name__=='__main__':
    df = pd.read_csv("data\\000001.SH_dataset.csv",index_col=0,parse_dates=['date'])
    delete_list = ['ts_code', 'trade_date', 'date']
    #main()
    t1 = time.clock()

    # one fixed window: classification and regression
    pred1 = np.array([])
    pred2 = np.array([])
    for i in create_window(df,60,1360,1300,'fixed one'):
        print(i.shape,i.index)
        dt_i = fe.process_feature(i,delete_list=delete_list,label_name="label",lags=5,begin_ind=0,end_ind=1000,gen_var=True,drop_var=False,ut=0.05,lt=-0.05)
        print(dt_i.columns,dt_i.shape) # debug
        y_pred1, in_acc, oos_acc, model1, cvresult1 = XGB.train_xgb(dt_i, "label", 0, 1000, 200, 5,True,
                                                                                     'multi:softprob', 1, 6, 0.5, 0.1, 5, 3,
                                                                                     500, 10, True)
        pred1 = np.append(pred1,y_pred1)


        y_pred2, in_mae, in_mse, oos_mae, oos_mse, model2, cvresult2 = XGB.train_xgb(dt_i, "close", 0, 1000, 200, 1,True,
                                                                                 'reg:linear', 1, 6, 0.5, 0.1, 5, 3,
                                                                                 500, 10, True)
        pred2 = np.append(pred2, y_pred2)
    plot_result(df, 60, 1065, 1265, pred1, "label", "fixed_one_xgb")
    plot_result(df, 60, 1062, 1262, pred2, "close", "fixed_one_xgb")

    # rolling window
    pred3 = np.array([])
    pred4 = np.array([])
    in_acc_list = []
    oos_acc_list = []
    in_mae_list = []
    in_mse_list = []

    ## classification
    for i in create_window(df, 60, 1360, 125, 'rolling',20):
        dt_i = fe.process_feature(i, delete_list=delete_list, label_name="label", lags=5, begin_ind=0, end_ind=100,
                                  gen_var=True, drop_var=False, ut=0.05, lt=-0.05)
        y_pred3, in_acc, oos_acc, model3, cvresult3 = XGB.train_xgb(dt_i, "label", 0, 100, 20, 5, True,
                                                                    'multi:softprob', 1, 6, 0.5, 0.1, 5, 3,
                                                                    500, 10, False)
        pred3 = np.append(pred3, y_pred3)
        oos_acc_list.append(oos_acc)
        in_acc_list.append(in_acc)
    print('in sample accuracy：%.2f%%' % (sum(in_acc_list)/len(in_acc_list) * 100))
    print('out of sample accuracy：%.2f%%' % (sum(oos_acc_list)/len(oos_acc_list) * 100))

    ## regression
    for i in create_window(df, 60, 1360, 121, 'rolling', 20):
        dt_i = fe.process_feature(i, delete_list=delete_list, label_name="close", lags=5, begin_ind=0, end_ind=100,
                                  gen_var=True, drop_var=False, ut=0.05, lt=-0.05)
        y_pred4, in_mae, in_mse, _, _, model4, cvresult4 = XGB.train_xgb(dt_i, "close", 0, 100, 20, 1,
                                                                                     True,
                                                                                     'reg:linear', 1, 6, 0.5, 0.1, 5, 3,
                                                                                     500, 10, False)
        pred4 = np.append(pred4, y_pred4)
        in_mae_list.append(in_mae)
        in_mse_list.append(in_mse)
    y_test = df["close"][162:1342]
    oos_mae = mean_absolute_error(y_test, pred4)
    oos_mse = mean_squared_error(y_test, pred4)
    print('in sample mean absolute error：%.2f' % (sum(in_mae_list) / len(in_mae_list)))
    print('in sample mean squared error：%.2f' % (sum(in_mse_list) / len(in_mse_list)))
    print('out of sample mean absolute error：%.2f' % (oos_mae))
    print('out of sample mean squared error：%.2f' % (oos_mse))
    plot_result(df, 60, 165, 1345, pred3, "label", "rolling_xgb")
    plot_result(df, 60, 162, 1342, pred4, "close", "rolling_xgb")

    # recursive window
    pred5 = np.array([])
    pred6 = np.array([])
    in_acc_list = []
    oos_acc_list = []
    in_mae_list = []
    in_mse_list = []

    ## classification
    for i in create_window(df, 60, 1360, 125, 'recursive', 20):
        train_len = len(i)-5-20
        print("length of training data: ",train_len)
        dt_i = fe.process_feature(i, delete_list=delete_list, label_name="label", lags=5, begin_ind=0, end_ind=train_len,
                                  gen_var=True, drop_var=False, ut=0.05, lt=-0.05)
        y_pred5, in_acc, oos_acc, model5, cvresult5 = XGB.train_xgb(dt_i, "label", 0, train_len, 20, 5, True,
                                                                    'multi:softprob', 1, 6, 0.5, 0.1, 5, 3,
                                                                    500, 10, False)
        pred5 = np.append(pred5, y_pred5)
        oos_acc_list.append(oos_acc)
        in_acc_list.append(in_acc)
    print('in sample accuracy：%.2f%%' % (sum(in_acc_list) / len(in_acc_list) * 100))
    print('out of sample accuracy：%.2f%%' % (sum(oos_acc_list) / len(oos_acc_list) * 100))
    # plot the oos acc of every window
    plt.plot(list(range(100, 1261, 20)), oos_acc_list,color='b')
    plt.xlabel("length of training set")
    plt.ylabel("out of sample accuracy on the test set")
    mean_acc = sum(oos_acc_list)/len(oos_acc_list)
    plt.axhline(mean_acc,linestyle='--',color='r',label="mean accuracy %.2f"%(mean_acc))
    plt.legend()
    plt.show()

    ## regression
    for i in create_window(df, 60, 1360, 121, 'recursive', 20):
        train_len = len(i) - 1 - 20
        print("length of training data: ", train_len)
        dt_i = fe.process_feature(i, delete_list=delete_list, label_name="close", lags=5, begin_ind=0, end_ind=train_len,
                                  gen_var=True, drop_var=False, ut=0.05, lt=-0.05)
        y_pred6, in_mae, in_mse, _, _, model6, cvresult6 = XGB.train_xgb(dt_i, "close", 0, train_len, 20, 1,
                                                                         True,
                                                                         'reg:linear', 1, 6, 0.5, 0.1, 5, 3,
                                                                         500, 10, False)

        pred6 = np.append(pred6, y_pred6)
        in_mae_list.append(in_mae)
        in_mse_list.append(in_mse)
    y_test = df["close"][162:1342]
    oos_mae = mean_absolute_error(y_test, pred6)
    oos_mse = mean_squared_error(y_test, pred6)
    print('in sample mean absolute error：%.2f' % (sum(in_mae_list) / len(in_mae_list)))
    print('in sample mean squared error：%.2f' % (sum(in_mse_list) / len(in_mse_list)))
    print('out of sample mean absolute error：%.2f' % (oos_mae))
    print('out of sample mean squared error：%.2f' % (oos_mse))
    plot_result(df, 60, 165, 1345, pred5, "label", "recursive_xgb")
    plot_result(df, 60, 162, 1342, pred6, "close", "recursive_xgb")

    t2 = time.clock()
    print("total time consumed: ".format(t2-t1))
  


