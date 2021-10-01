import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_bias(df,n):
    """
    Calculate the n days bias ratio
    INPUT
    df: a dataframe
    n: the number of days simple average is calculated of
    OUTPUT
    bias pandas series
    INTERPRETATION
    bigger the value, more extreme today's price is compared to a recent average
    """
    ts = df.close
    rolling_avg = ts.rolling(window = n, min_periods = n).mean() # simple average of n days
    bias = (ts-rolling_avg)/rolling_avg*100
    return bias


def calc_CR(df,n):
    """
    Calculate the CR ratio
    INPUT
    df: a dataframe
    n: the number of days simple average is calculated of
    OUTPUT
    CR pandas series
    INTERPRETATION
    bigger the value, stronger the force of buy-in rather than sell-out
    """
    mid = (df.high.shift(1) + df.low.shift(1))/2
    ascend = df.high - mid #rising value
    ascend[ascend<=0] = 0
    descend = mid - df.low #falling value
    descend[descend<=0] = 0
    long_intensity = ascend.rolling(window=n, min_periods=n).sum()
    short_intensity = descend.rolling(window=n, min_periods=n).sum()
    CR = long_intensity/short_intensity * 100
    return CR

def calc_ROC(df,n):
    """
    Calculate the n-day change ratio of price
    INPUT
    df: a dataframe
    n: the number of days ahead that is the compared baseline date
    OUTPUT
    ROC pandas series
    INTERPRETATION
    bigger the value, more distinct the price is compared to the hypothesized one period ahead
    """
    BX = df.close.shift(n)
    AX = df.close - BX 
    ROC = AX/BX*100
    return ROC 

def calc_single_day_VPT(df):
    """
    Calculate volumn price trend
    INPUT
    df: a dataframe
    OUTPUT
    VPT pandas series
    """
    VPT = (df.close - df.close.shift(1))/df.close.shift(1)*df.vol
    return VPT

def rank(df,n):
    """
    Calculate the current price's rank in the past n days
    INPUT
    df: a dataframe
    n: number of lags considered
    OUTPUT
    rank pandas series
    """
    length = len(df)
    rank = []
    for i in range(length):
        if i < n:
            rank.append(np.nan)
        else:
            past_price = df.close[i-n:i].sort_values(ignore_index=True)
            diff_price = past_price - df.close[i]
            diff_price_abs = diff_price.abs()
            r = diff_price_abs[diff_price_abs==diff_price_abs.min()].index[0] + 1
            rank.append(r)
    return rank

def cal_VROC12(df,n=12):
    """
    Calculate the n-day change ratio of value
    INPUT
    df: a dataframe
    n: the number of days ahead that is the compared baseline date
    OUTPUT
    VROC pandas series
    INTERPRETATION
    bigger the value, more distinct the volume is compared to the hypothesized one period ahead
    """

    BX = df.vol.shift(n)
    AX = df.vol - BX 
    VROC = AX/BX*100
    return VROC 

def cal_TVMA6(df,n=6):
    TVMA6=df.amount.rolling(window=n,min_periods=6).mean()
    return TVMA6
    
    
def cal_DAVOL(df,short,long):
    """
    Calculate the short-term avearge turnover rate VS long-term avearge turn over rate  
    INPUT
    df: a dataframe
    short: the time span of short term in days
    long: the time span of long term in days 
    OUTPUT
    A ratio of short-term turnover rate vs long-term turnover rate
    INTERPRETATION:
    bigger the value, more volatile in the way of turnover the stock is
    
    """
    short_ave=df.turnover_rate_f.rolling(window=short,min_periods=short).mean()
    long_ave=df.turnover_rate_f.rolling(window=long,min_periods=long).mean()
    DAVOL=short_ave/long_ave
    return DAVOL


def cal_WVAD6(df,n=6):
    """
    Calculate the volatility within a day
    INPUT
    df: a dataframe
    n:the time span of summation
    OUTPUT
    series
    INTERPRETATION:
    bigger the value, more volatile in the way of amount the stock is
    
    """
    series=(df.close-df.open)/(df.high-df.low)*df.vol
    return series.rolling(window=n,min_periods=n).sum()


def cal_VSTD20(df,n=20):
    """
    Calculate the volatility of volume
    INPUT
    df: a dataframe
    n: calculating period in days
    OUTPUT
    series
    INTERPRETATION:
    bigger the value, more volatile the stock is
    
    """
    return df.vol.rolling(window=n,min_periods=n).std()



def cal_AR(df,n=26):
    """
    calculate the popularity of this stock
    """
    nominator=(df.high-df.open).rolling(window=n,min_periods=n).sum()
    denominator=(df.open-df.low).rolling(window=n,min_periods=n).sum()
    return nominator/denominator*100


def cal_VOL5(df,n=5):
    """
    5-day average turnover rate of free shares
    """
    return df.turnover_rate_f.rolling(window=n,min_periods=n).mean()
    
    
def turnover_volatility20(df,n=20):
    """
    volatility of turnover rate of rfree shares in a 20-day time span
    """
    return df.turnover_rate_f.rolling(window=n,min_periods=n).std()

def cal_BR(df,n=26):
    """willingness indicator,pairing with AR"""
    nominator=(df.high-df.close.shift(1)).rolling(window=n,min_periods=n).sum()
    denominator=(df.close.shift(1)-df.low).rolling(window=n,min_periods=n).sum()
    return nominator/denominator*100

def cal_ARBR(df,n=26):
    return df["AR26"]-df["BR26"] 

def moneyflow20(df,n=20):
    "20-day average moneyflow"
    one_day_moneyflow=(df.open+df.close)/2*df.vol
    ave_moneyflow20=one_day_moneyflow.rolling(window=n,min_periods=n).mean()
    return  ave_moneyflow20


def cal_var(df, n):
    """
    Calculate the n-day annual yield variance
    INPUT
    df: a dataframe
    n:the time span
    OUTPUT
    series
    INTERPRETATION:
    bigger the value, more volatile the annual yield is
    """
    one_day_annual_yield = np.log(df.close/df.open) * 365
    var_annual_yield_20 = one_day_annual_yield.rolling(window=n, min_periods=n).var()
    return var_annual_yield_20


def cal_skew(df, n):
    """
    Calculate the n-day yield skewness
    INPUT
    df: a dataframe
    n:the time span
    OUTPUT
    series
    INTERPRETATION:
    Describe the distribution of the yield
    """
    one_day_yield = np.log(df.close/df.open)
    skew = one_day_yield.rolling(window=n, min_periods=n).skew()
    return skew


def cal_kurtosis(df, n):
    """
    Calculate the n-day yield kurtosis
    INPUT
    df: a dataframe
    n:the time span
    OUTPUT
    series
    INTERPRETATION:
    Describe the distribution of the yield
    """
    one_day_yield = np.log(df.close/df.open)
    kurtosis = one_day_yield.rolling(window=n, min_periods=n).kurt()
    return kurtosis


def cal_sharpe_ratio(df, n=20):
    """
    Calculate the 20-day sharpe ratio
    INPUT
    df: a dataframe
    n:the time span
    OUTPUT
    series
    INTERPRETATION:
    Describe the ratio of return and risk
    """
    one_day_annual_yield = np.log(df.close/df.open) * 365
    sharpe_ratio = (one_day_annual_yield - 0.04)/one_day_annual_yield.rolling(window=n, min_periods=n).std()
    return sharpe_ratio

def cal_logret(df):
    """
    calculate the logarithmic return
    """
    return np.log(df.close/df.close.shift(1))

#triple barrier 
def mark_Y(df,label_var,t,upper_r,lower_r):
    """
    triple barrier labelling
    INPUT
    t: int, time barrier parameter
    label_var：string, the variable's name, of which the label is based on
    upper_r: float, gain barrier parameter
    lower_r: float, loss barrer parameter
    """
    Ys=[]
    for i in range(len(df)-t):#只有N-t天的数据
        flag=0
        now=0
        while now<t:#纵向屏障，时间限制
            if df[label_var][i+now]>=(1+upper_r)*df[label_var][i]:#上界收益限制
                flag=1
                break
            elif df[label_var][i+now]<=(1-lower_r)*df[label_var][i]:#下届收益限制
                flag=-1
                break
            else:
                now+=1
        Ys.append(flag)

    for i in range(len(df)-t,len(df)):
        Ys.append(np.nan)
    return Ys

def get_date(a):
    """
    transform the trade_date column to pandas datetime object
    """
    date = str(a)
    date = date[:4]+'/'+date[4:6]+'/'+date[6:]
    return date

def triple_barrier_plot(df,file_name):
    """
    Plot the time series of closing price and its labelling
    """
    # df = df.dropna(subset=['label'],axis=0)
    df["date"] = df.trade_date.apply(lambda x: get_date(x))
    df["date"] = pd.to_datetime(df["date"])

    plt.style.use('ggplot')
    fig1 = plt.figure(1,figsize=(16,9))
    colors = ['g','r','b']
    Label_Com = ['up','down','oscillate']
    label_list = [1,-1,0]
    for item in label_list:
        Price = df.loc[df['label'] == item]['close']
        Index = df.loc[df['label'] == item]['date']
        index = label_list.index(item)
        plt.scatter(Index, Price, c=colors[index], cmap='brg', s=30, alpha=0.3, marker='8', linewidth=0)

    ax = fig1.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    #added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels = Label_Com, loc='upper right')

    ax2 = ax.twinx()
    ax2.plot(df.date,df.close,c='b',alpha=0.2)

    plt.show()
    plt.savefig("out\\figs\\{}.jpg".format(file_name))
    plt.close()


def create_dataset(dataset_name,plot_name):
    df = pd.read_csv("data\\{}.csv".format(dataset_name), index_col=0)
    df.sort_values(by="trade_date", inplace=True)

    SPX = pd.read_csv("data\\SPX.csv", index_col=0)  # 标普500指数
    XIN9 = pd.read_csv("data\\XIN9.csv", index_col=0)  # 富时中国A50指数
    HSGT = pd.read_csv("data\\HSGT.csv", index_col=0)  # 沪深港通数据
    NHCI = pd.read_csv("data\\NHCI.NH.csv", index_col=0)  # 南华商品指数

    """
       动量指标 Momentum Factors

       """
    # 乖离率
    df["bias5"] = calc_bias(df, 5)
    df["bias10"] = calc_bias(df, 10)
    df["bias20"] = calc_bias(df, 20)
    df["bias60"] = calc_bias(df, 60)

    # 多方强度/空房强度CR指标
    df["CR20"] = calc_CR(df, 20)

    # 12日价格变动率
    df["ROC12"] = calc_ROC(df, 12)

    # 单日价量趋势
    df["VPT"] = calc_single_day_VPT(df)

    # 60日价格序位
    df["rank60d"] = rank(df, 60)

    """
    情绪因子 Sentiment Factors

    """
    # 12日交易量变化率
    df["VROC12"] = cal_VROC12(df)

    # 6日交易额均值
    df["TVMA6"] = cal_TVMA6(df)

    # 五日换手率均值/60日换手率均值
    df["DAVOL5_60"] = cal_DAVOL(df, 5, 60)

    # 6日威廉变异离散率
    df["WVAD6"] = cal_WVAD6(df)

    # 20日成交量标准差
    df['VSTD20'] = cal_VSTD20(df)

    # 26日人气指标
    df['AR26'] = cal_AR(df)

    # 26日意愿指标
    df["BR26"] = cal_BR(df)

    # 5日平均换手率
    df["VOL5"] = cal_VOL5(df)

    # 20日换手率波动率
    df["turnover_volatility20"] = turnover_volatility20(df)

    # 人气指标与意愿指标之差
    df["ARBR"] = cal_ARBR(df)

    # 20日平均资金流量
    df["moneyflow20"] = moneyflow20(df)

    """
    risk factors
    """
    # 20-day annual interest rate variance
    df["variance20"] = cal_var(df, 20)

    # 20-day stock skewness
    df["skewness20"] = cal_skew(df, 20)

    # 20-day stock kurtosis
    df["kurtosis20"] = cal_kurtosis(df, 20)

    # 20-day sharpe_ratio
    df["sharpe20"] = cal_sharpe_ratio(df, 20)

    # 60-day annual interest rate variance
    df["variance60"] = cal_var(df, 60)

    # 60-day stock skewness
    df["skewness60"] = cal_skew(df, 60)

    # 60-day stock kurtosis
    df["kurtosis60"] = cal_kurtosis(df, 60)

    # logarithmic return
    df["log_ret"] = cal_logret(df)

    # Triple barrier labelling
    df["label"] = mark_Y(df, "close", 5, 0.01, 0.01)

    # 南华商品指数
    df = pd.merge(df, NHCI[["trade_date", "close", "pct_chg"]], suffixes=('', '_NHCI'), how="left", on="trade_date")
    df.rename(columns={"pct_chg": "NHCI_pct_chg"}, inplace=True)

    # 标普500指数
    df = pd.merge(df, SPX[["trade_date", "close", "pct_chg"]], suffixes=('', '_SPX'), how="left", on="trade_date")
    df.rename(columns={"pct_chg": "SPX_pct_chg"}, inplace=True)

    # 富时中国A50指数
    df = pd.merge(df, XIN9[["trade_date", "close", "pct_chg"]], suffixes=('', '_XIN9'), how="left", on="trade_date")
    df.rename(columns={"pct_chg": "XIN9_pct_chg"}, inplace=True)

    # 沪深港通数据
    df = pd.merge(df, HSGT[["trade_date", "north_money", "south_money"]], how="left", on="trade_date")

    df[["close_NHCI", "NHCI_pct_chg", "close_SPX", "SPX_pct_chg", "close_XIN9", "XIN9_pct_chg", "north_money",
        "south_money"]] = df[
        ["close_NHCI", "NHCI_pct_chg", "close_SPX", "SPX_pct_chg", "close_XIN9", "XIN9_pct_chg", "north_money",
         "south_money"]].fillna(method='ffill')
    print(df.head())

    # plot the closing price and its labelling
    triple_barrier_plot(df,plot_name)

    df.to_csv("data\\{}_dataset.csv".format(dataset_name))

if __name__ == '__main__':
    create_dataset("000001.SH","000001.SH_label")
    create_dataset("000016.SH", "0000161.SH_label")
    create_dataset("000905.SH", "000905.SH_label")
    create_dataset("399001.SZ", "399001.SZ_label")

