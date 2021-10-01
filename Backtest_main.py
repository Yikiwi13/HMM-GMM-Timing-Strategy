import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from Backtest import *
from create_dataset import mark_Y

"""
conduct backtest on signals formed on predictions of the recursive XGBoost regression
"""

# load data
df = pd.read_csv("data\\000001.SH_dataset.csv",index_col=0,parse_dates=['date'])
y_test = df["close"][162:1342]
date = df["date"][162:1342]
open = df["open"][162:1342]

def test(y_pred,label):
    global y_test,date,open
    df_y = pd.DataFrame({'date':date,'close':y_test,'open':open,'pred':y_pred}).reset_index()
    df_y['label_pred'] = mark_Y(df_y, "pred", 5, 0.01, 0.01)
    df_y['label_true'] = mark_Y(df_y, "close", 5, 0.01, 0.01)

    # inspecting the accuracy of the three category classification
    acc_total = (df_y['label_pred'] == df_y['label_true']).sum()/(len(df_y)-5) # total accuracy
    acc_buy = ((df_y['label_pred']==1)&(df_y['label_true']==1)).sum()/(df_y['label_true']==1).sum() # accuracy when the true label == 1
    acc_sell = ((df_y['label_pred']==-1)&(df_y['label_true']==-1)).sum()/(df_y['label_true']==-1).sum() # accuracy when the true label == -1
    acc_osci = ((df_y['label_pred']==0)&(df_y['label_true']==0)).sum()/(df_y['label_true']==0).sum() # accuracy when the true label == 0

    # backtest
    b = backtest(0.0005,0.002)
    b.set_money(10000)
    values = []

    for i in range(0,len(df_y)-5):
        price = df_y['open'][i] # the transaction is made at the market opening
        signal = df_y['label_pred'][i]
        if signal == 1:
            b.buy(price)
            values.append(b.value)
        if signal == -1:
            b.sell(price)
            values.append(b.value)
        if signal == 0:
            values.append(b.value)
    print(values)

    buy_hold = 10000/df_y['open'][0]*df_y['open'][:-5]

    date = df_y['date'][:-5]
    curve = pd.DataFrame({'date':date,'value':values,'benchmark':buy_hold})

    # plot
    fig1 = plt.figure(figsize=(16, 9))
    plt.plot(curve.date,curve.value,label=label)
    plt.plot(curve.date,curve.benchmark,label='buy and hold')
    plt.legend()
    plt.show()

    # ret
    curve['ret'] = (curve['value']-curve['value'].shift(1))/curve['value']
    mkt_ret = (buy_hold - buy_hold.shift(1))/buy_hold
    sr = b.calc_sharpeRatio(curve['ret'],mkt_ret)
    print('sharpe ratio: %.2f' % sr)

    maxd,end_ind,begin_ind = b.calc_maxDrawdown(curve.value)
    begin_date,end_date = curve.date[begin_ind],curve.date[end_ind]
    print('max_drawdown: %.2f%%' % (maxd*100))
    print("begin date: ",begin_date,"  end date: ",end_date)

    result = {'curve':curve,'max_drawdown':(maxd,begin_date,end_date),'sharpe ratio':sr,'acc':(acc_total,acc_buy,acc_sell,acc_osci)}
    return result

if __name__=='__main__':
    # y_pred = pkl.load(open("out\\results\\xgb_reg_re.pkl",'rb'))
    y_pred = pd.read_pickle("out\\results\\xgb_reg_re.pkl")
    result = test(y_pred,'XGBoost regressor (recursive window)')
    #pkl.dump(result,open("out\\results\\backtest_reg_re.pkl",'wb'))


