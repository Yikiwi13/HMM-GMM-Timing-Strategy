"""
correlation analysis of features & generation of lags of existing features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import time

def corr_analysis(df,delete_list,label_name,lags,begin_ind=0,end_ind=-4,gen_var=False,ut=0.0,lt=0.0,min_lag=1,plot=False,save=False):
    """
    correlation analysis
    INPUT
    df: dataset
    delete_list: features not involved in the correlation analysis
    label_name: name of the label columns
    lags: lags of features considered in the correlation analysis
    begin_ind: begin index of the dataset that is included in correlation analysis
    end_ind: end index of the dataset that is included in correlation analysis
    gen_var: boolean indicator of whether lags variables shall be created, default False
    ut: upper threshold, if the coefficient between a lagged variable and the label >= ut, the variable will be considered
    lt: lower threshold
    min_lag: if lag >= min_lag, lagged variable will created,default 1
    plot: whether to plot the correlation matrices or not, default false
    save: whether to save the correlation matrices or not, default false
    OUTPUT
    correlation matrices,
    correlation heatmap plots,
    new_df: lags variables concatenated to the dataset df
    """
    new_df = df.copy()
    dt = df.drop(columns = delete_list)


    for i in range(lags+1):
        dt['label_'] = dt[label_name].shift(-i)
        corr_mat = dt.iloc[begin_ind:end_ind, :].corr() # only the training set's corr_mat shall be known
        #print(dt.iloc[begin_ind:end_ind, :].shape) # debug

        if plot:
            f, ax = plt.subplots(figsize=(10, 10))
            plt.title('correlation analysis, lag {}'.format(i))
            sns.heatmap(corr_mat, annot=False, Linewidths=0.2, annot_kws={'size': 10})
            f.savefig('out\\figs\\correlation analysis, lag {}.jpg'.format(i), dpi = 500, bbox_inches = 'tight')
            plt.show()

        if save:
            corr_mat.to_csv("out\\results\\corr_mat_lag{}.csv".format(i))
            # pkl.dump(corr_label,open("out\\results\\corr_label_lag{}.pkl".format(i),"wb"))

        if gen_var:
            new_df = gen_lagged_var(corr_mat,new_df,ut,lt,i,min_lag)

    return new_df


def gen_lagged_var(corr_mat,df,ut,lt,lag,min_lag=1):
    """
    generate lags of feature variables based on correlation analysis of the training set
    INPUT
    corr_mat: correlation matrix
    ut: upper threshold, if the coefficient between a lagged variable and the label >= ut, the variable will be considered
    lt: lower threshold
    lag: lag of the feature
    min_lag: if lag >= min_lag, lagged variable will created,default 1
    drop_var: when lag=0, drop existing feature variables that has lower coefficient (in between lt and ut) with the target variable
    OUTPUT
    lags variables concatenated to the dataset df
    """
    dt = df.copy()
    corr_mat.drop(labels=["label_"], axis=0, inplace=True)

    if lag<4:
        corr_mat.drop(labels=["label"],axis=0,inplace=True)

    corr_label = pd.DataFrame(corr_mat.label_)
    corr_label = corr_label[(corr_label.label_ >= ut) | (corr_label.label_ <= lt)]
    feature_list = corr_label.index.to_list()

    if min_lag<=lag:
        for feature in feature_list:
            name = feature+'_lag_'+str(lag)
            dt[name] = df[feature].shift(lag)
    return dt

def dropvar(df,delete_list,label_name,ut=0.0,lt=0.0):
    """
    INPUT
    df: dataset
    delete_list: features not involved in the correlation analysis
    label_name: name of the label columns
    ut: upper threshold, if the coefficient between a lagged variable and the label >= ut, the variable will not be dropped
    lt: lower threshold
    """
    dt = df.copy()
    df = df.drop(columns=delete_list)
    corr_mat = df.corr()
    corr_label = pd.DataFrame(corr_mat[label_name])
    corr_label = corr_label[(corr_label[label_name] < ut) & (corr_label[label_name] > lt)]
    delete_list = corr_label.index.to_list()
    dt.drop(columns=delete_list, inplace=True)
    return dt

def process_feature(df, delete_list, label_name, lags,begin_ind=0,end_ind=-4,gen_var=False, drop_var=False,ut=0.0, lt=0.0, drop_ut=0.0,drop_lt=0.0,min_lag=1, plot=False, save=False):
    """
    procss feature, generate lagged variables or delete variables according to its correlation with the target variable
    INPUT
    dt: pandas dataframe, the raw dataset
    delete_list: list, features not involved in the correlation analysis
    label_name: string, name of the label columns
    lags: int, lags of features considered in the correlation analysis
    begin_ind: begin index of the dataset that is included in correlation analysis
    end_ind: end index of the dataset that is included in correlation analysis
    gen_var: bool, boolean indicator of whether lags variables shall be created, default False
    drop_var: bool, boolean indicator of whether variables with lower coefficient shall be dropped, default False
    ut: float, upper threshold, if the coefficient between a lagged variable and the label >= ut, the variable will be considered
    lt: float, lower threshold
    drop_ut: float, upper threshold, to decide whether a variable shall be dropped or not
    drop_lt: float, lower threshold
    min_lag: int, if lag >= min_lag, lagged variable will created,default 1
    plot: bool, whether to plot the correlation matrices or not, default false
    save: bool, whether to save the correlation matrices or not, default false
    OUTPUT
    correlation matrices, csv files
    correlation heatmap plots, jpg files
    new_df: dataset after adding or drop variables, pandas dataframe
    """
    new_df = corr_analysis(df, delete_list, label_name, lags, begin_ind,end_ind,gen_var, ut, lt, min_lag, plot, save)
    if drop_var:
        new_df = dropvar(new_df,delete_list,label_name,drop_ut,drop_lt)
    return new_df



if __name__=="__main__":
    """
    df = pd.read_csv("data\\SH000001@0820.csv",index_col=0,parse_dates=['date'])
    delete_list = ['ts_code', 'trade_date', 'date']
    df1 = corr_analysis(df=df,delete_list=delete_list,label_name="label", lags=5, gen_var=True,ut=0.06,lt=-0.06,min_lag=1)
    df1 = drop_var(df=df,delete_list=delete_list,label_name="label",ut=0.03,lt=-0.03)
    df1.to_csv("data\\SH000001@0824_lags_drop.csv")
    df2 = corr_analysis(df=df,delete_list=delete_list,label_name="label", lags=5, gen_var=True,ut=0.06,lt=-0.06,min_lag=1)
    df2.to_csv("data\\SH000001@0824_lags.csv")

drop_var: when lag=0, drop existing feature variables that has lower coefficient (in between lt and ut) with the target variable

if (i == 0) and drop_var:
        corr_label = pd.DataFrame(corr_mat.label_)
        corr_label = corr_label[(corr_label.label_ >= ut) | (corr_label.label_ <= lt)]
        feature_list = corr_label.index.to_list()
        delete_list = list(set(corr_mat.index.to_list()) - set(feature_list))
        new_df.drop(columns=delete_list, inplace=True)
"""
    df = pd.read_csv("data\\SH000001@0820.csv", index_col=0, parse_dates=['date'])
    delete_list = ['ts_code', 'trade_date', 'date']
    df1 = process_feature(df, delete_list, "label", 5, begin_ind=60,end_ind=-5,gen_var=True, drop_var=False,ut=0.06, lt=-0.06, min_lag=1, plot=False, save=False)
    df2 = process_feature(df, delete_list, "label", 5, begin_ind=60,end_ind=-5, gen_var=True, drop_var=True, ut=0.06,lt=-0.06, drop_ut=0.03,drop_lt=-0.03, min_lag=1, plot=False, save=False)
    #df1.to_csv("data\\SH000001@0825_lags.csv")
    #df2.to_csv("data\\SH000001@0825_lags_drop.csv")
