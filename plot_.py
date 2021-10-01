import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
from create_dataset import mark_Y,triple_barrier_plot

"""
using triple barrier labelling to convert the regressions' predictions to signals of "up", "down", and "oscillate" 
"""
df = pd.read_csv("data\\000001.SH_dataset.csv",index_col=0,parse_dates=['date'])
pred4 = pkl.load(open("out\\results\\xgb_reg_rw.pkl",'rb')) #xgb regression rolling window
pred6 = pkl.load(open("out\\results\\xgb_reg_re.pkl",'rb')) #xgb regression recursive window

df1 = df.iloc[162:1342,:]
df1 = df1.reset_index()
df1["pred4"] = pred4
df1["pred6"] = pred6

# generate signals using the triple barrier method
df1["label_rolling"] = mark_Y(df1, "pred4", 5, 0.01, 0.01)
df1["label_recursive"] = mark_Y(df1, "pred6", 5, 0.01, 0.01)
df1 = df1.dropna(subset=["label_rolling","label_recursive"],axis=0)
acc_rolling = sum(df1["label_rolling"]==df1["label"])/len(df1)
acc_recursive = sum(df1["label_recursive"]==df1["label"])/len(df1)

print('out of sample accuracy (rolling window regression)：%.2f%%' % (acc_rolling*100))
print('out of sample (recursive window regression)：%.2f%%' % (acc_recursive*100))
df1["label"] = df1["label_rolling"]
triple_barrier_plot(df1,"XGB_reg_rw_tbplot")
df1["label"] = df1["label_recursive"]
triple_barrier_plot(df1,"XGB_reg_re_tbplot")
"""
fig1 = plt.figure(figsize=(16, 9))
plt.plot(df["date"][60:1261], df["close"][60:1261], c='b', alpha=0.2, label='realized')
plt.scatter(df2["date"], df2["pred"], c='b', s=20, alpha=0.3, marker='8', linewidth=0,label='predicted')
plt.xlabel("Date")
#plt.xlim(xmin=df["date"][begin_ind], xmax=df["date"][end_ind])
plt.ylim(ymin=2000, ymax=5000)
plt.ylabel("Closing price")
plt.legend()
plt.show()
#plt.savefig("out\\figs\\{}predictClose.jpg".format(file_name))
#plt.close()
"""

