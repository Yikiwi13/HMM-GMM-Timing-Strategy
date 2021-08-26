"""
SECTION II DATA PROCESS---FEATURE SELECTION
version 2018:
    Liu et al., 2018. Stock Market Trend Analysis Using Hidden Markov
    Model and Long Short Term Memory.
version 2022:
    This section provides code that corresponds to the section II. DATA PROCESS of Liu et al. (2018).
    As the dataset has been changed, this program's result is not identical to the paper's.
    The code below is for your reference only, any comments, corrections and suggestions are highly appreciated.
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import time

def score_func(df,feature, n_states, begin_index, end_index):
    """
    feature selection function defined by eqn (1) to eqn (4) of Section II, page 2
    INPUT
        df: a pandas dataframe contains labels and the observation sequence
        feature: a string of the feature's name
        n_states: number of states in the hmm model
        begin_index: the index where the training set begins
        end_index: the index where the training set ends
    OUTPUT
        score of a given feature, as defined in eqn (1) to eqn (4) of Section II, page 2
    """
    X = df[feature][begin_index:end_index]
    X = np.array(X).reshape(-1,1)
    model = hmm.GMMHMM(n_components=n_states).fit(X) # fit the training sequence
    Y = model.predict(X) # predictions of the training sequence
    label = df["label"][begin_index:end_index]
    result = pd.DataFrame({"states": Y, "label": label})
    M = np.zeros((n_states, 3)) # count matrix M
    for s in range(n_states):
        for l in range(3):
            M[s,l] = len(result[(result.states==s)&(result.label==l-1)])
    MR = M.copy()

    score_list = [] # score component of a specific state
    for s in range(n_states):
        row_sum = np.sum(M[s,:])
        MR[s,:] = np.divide(M[s,:],row_sum) # count ratio matrix MR
        ind = np.argmax(MR[s,:])
        acc = MR[s,ind] # eqn (1), of Section II, page 2
        ent = - MR[s,:]*np.log(MR[s,:]) # eqn (2), of Section II, page 2
        w = np.sum(M[s,:])/np.sum(M) # eqn (3), of Section II, page 2
        score_s = acc * (1/(1+ent))*w # eqn (4), of Section II, page 2
        score_list.append(score_s)
    score = np.sum(score_list) # eqn (4), of Section II, page 2
    return score

if __name__=="__main__":
    start = time.clock()

    df = pd.read_csv("data\\SH000001@0820.csv",index_col=0,parse_dates=['date'])
    feature_list = df.columns.to_list()
    delete_list = ['ts_code','trade_date','date','label']
    for item in delete_list:
        feature_list.remove(item)
    scores = []
    for feature in feature_list:
        score = score_func(df,feature, 3, 60, 1000)
        scores.append(score)

    result = pd.DataFrame({"feature":feature_list,"score":scores})
    result.sort_values(by='score',inplace=True)
    result.to_csv("out\\results\\section_II_score_function.csv")

    plt.figure(figsize=(16, 9))
    plt.bar(result.feature, result.score)
    plt.xticks(result.feature, rotation=60)
    plt.xlabel("features")
    plt.ylabel("scores")
    plt.savefig("out\\figs\\fig3_feature_scores.jpg",dpi=500,bbox_inches='tight')
    plt.show()

    end = time.clock()
    print(end-start, "seconds")



