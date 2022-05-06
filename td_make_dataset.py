from config import ACCT_NUMBER, API_KEY, CALLBACK_URL, JSON_PATH
from td.client import TDClient
import pprint
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#connecting to tdameritrade w/ credentials
td_client = TDClient(client_id = API_KEY, redirect_uri = CALLBACK_URL, account_number = ACCT_NUMBER, credentials_path = JSON_PATH)
td_client.login()

#establish parameters to pull stock data in a specified range from start date to present
# datetime(year, month, day, hour, minute, second, microsecond)
howfarback = 365
# present_date = datetime.now()
# past_date = datetime.now() - timedelta(days = howfarback)
present_date = datetime(2017, 12, 31, 23, 59, 59, 999999)
past_date = datetime(2017, 1, 1, 00, 00, 000000)
present_date = str(int(round(present_date.timestamp() * 1000))) #convert datetime to ms
past_date = str(int(round(past_date.timestamp() * 1000)))

#formulate request
daily_data = td_client.get_price_history(symbol = 'AAPL', 
                                          period_type = 'month',
                                          frequency_type = 'daily',
                                          frequency = 1, 
                                          start_date = past_date,
                                          end_date = present_date, 
                                          extended_hours = True)

#create dataframe
df_daily = pd.DataFrame(daily_data['candles'])

#convert datetime(ms) to date format: YYYY-MM-DD
df_dates = []
for day in df_daily['datetime']:
    day = datetime.fromtimestamp(int(day)/1000).strftime('%Y-%m-%d')
    df_dates.append(day)
df_daily['date'] = df_dates #add column

#set date format as index and drop datetime(ms) & duplicated date columns
df_daily = df_daily.set_index(pd.DatetimeIndex(df_daily['date'])) \
    .drop(columns = ['datetime', 'date']).rename_axis('Date', axis=1)
print(df_daily)
#function for Simple Moving Average (SMA)
def SMA(data, period, col = 'close'):
    return data[col].rolling(window = period).mean()

#function for Exponential Moving Average (EMA)
def EMA(data, period, col = 'close'):
    sma_data = SMA(data, period)[:period]
    leftover_data = data[col][period:]
    return pd.concat([sma_data, leftover_data]).ewm(span = period, adjust=False).mean()

#function for identifying what type of MA pair calculation to evaluate 
def hybrid_MA(MA, data, period, col = 'close'):
    if MA == 'SMA':
        return SMA(data, period, col = 'close')
    elif MA == 'EMA':
        return EMA(data, period, col = 'close')
        
#all SMA periods split by the short, mid, long term periods (***253 trading days per year)
SMA_short = list(range(4, 43)) #range: 4 days-2 months
SMA_mid = list(range(63, 127)) #range: 3 months-6 months
SMA_long = list(range(190, 254)) #range: 9 months-1 year

#function to return an ordered list in a dict of the most profitable MA strategy given the input: MA_pair
def MA_buysell_indicators(MA_pair, data):    
    
    #transforming the str of MA pairs to identify which period and which MA calculation to use
    pair = MA_pair.split('__')
    if pair[0][0] == 's':
        MA1 = SMA_short
    elif pair[0][0] == 'm':
        MA1 = SMA_mid
    if pair[1][0] == 's':
        MA2 = SMA_short
    elif pair[1][0] == 'm':
        MA2 = SMA_mid
    if pair[0][1] == 'S':
        MA_func = 'SMA'
    elif pair[0][1] == 'E':
        MA_func = 'EMA'
    if pair[1][1] == 'S':
        MA_func = 'SMA'
    elif pair[1][1] == 'E':
        MA_func = 'EMA'
        
    #iterating through all combinations
    p1_list, p2_list, r_profits, buysell_prices, buysell_dates = ([] for lst in range(5))
    for p1 in MA1:
        for p2 in MA2:
            df = data.copy()
            name_MA1 = pair[0] + str(p1)
            name_MA2 = pair[1] + str(p2)
            df[name_MA1] = hybrid_MA(MA_func, df, p1)
            df[name_MA2] = hybrid_MA(MA_func, df, p2)

            #generate the close price where the MA1 and MA2 cross
            df['signal'] = np.where(df[name_MA1] > df[name_MA2], 1, 0)
            df['position'] = df['signal'].diff()
            df['buy'] = np.where(df['position'] == 1, df['close'], 0)
            df['sell'] = np.where(df['position'] == -1, df['close'], 0)
            
            #dropping null values but keeping the original index
            buy_prices = [price for price in df['buy']]
            sell_prices = [price for price in df['sell']]
            buy = [list([i, n, 'b']) for i, n in enumerate(buy_prices) if n > 0]
            sell = [list([i, n, 's']) for i, n in enumerate(sell_prices) if n > 0]
            
            #joining the list into one ordered timeline based off the index(1st element)
            trades = buy + sell
            trades.sort()
            n_shares = 1 ###number of shares bought (implement even distribution or percentage distribution based off volatility/risk)
            
            #calculate and store returns/index based off combination
            trade_price = []
            trade_index = []
            for trade in trades:
                trade_index.append(trade[0])
                if trade[2] == 'b':
                    trade_price.append(round(trade[1]*-1*n_shares, 2))
                elif trade[2] == 's':
                    trade_price.append(round(trade[1]*n_shares, 2))
            returns = sum(trade_price)
            buysell_prices.append(trade_price)
            
            #convert date format and store the days of the buy/sell points
            bs_date = []
            for i in trade_index:
                _date = str(list(df.index.values)[i]).split('T')[0]
                bs_date.append(_date)
            
            #append the parameters to transpose onto new dataframe
            p1_list.append(p1)
            p2_list.append(p2)
            r_profits.append(returns)
            buysell_dates.append(bs_date)
            
            df = data.copy() #clears the combination's results for the next iteration
    
    MA_pair_df = pd.DataFrame({'dates': buysell_dates, 
                               'p_MA1': p1_list, 
                               'p_MA2': p2_list,
                               'buy/sell': buysell_prices,
                               'profit': r_profits}).sort_values(by = 'profit', ascending=False)

    return MA_pair_df

#list of all possible combinations for SMA vs EMA; s->short & m->mid
SMA_pairs = ['sSMA__sSMA', 'sSMA__mSMA', 'mSMA__mSMA']
hybridMA_pairs = ['sEMA__sSMA', 'sSMA__mEMA', 'sEMA__mSMA', 'mEMA__mSMA']
EMA_pairs = ['sEMA__sEMA', 'sEMA__mEMA', 'mEMA__mEMA']

#find max profit, top 50 mean, and  in a dataframe
def MA_profit_stats(dataframe):
    profit_details = pd.DataFrame({'pair_id':str(dataframe),
                                   'highest_profit':dataframe['profit'].max(),
                                   'mean_profit':dataframe['profit'].mean(),
                                   'std_dev':dataframe['profit'].std(),
                                   'n_std_frm_max':(dataframe['profit'].max() - dataframe['profit'].mean()) / profit_details['std_dev']}) #number of standard deviations b/e the mean and the highest profit
    
    return profit_details

def run_MA_func():
    sSMA__sSMA = MA_buysell_indicators(SMA_pairs[0], df_daily).head(50)
    sSMA__mSMA = MA_buysell_indicators(SMA_pairs[1], df_daily).head(50)
    mSMA__mSMA = MA_buysell_indicators(SMA_pairs[2], df_daily).head(50)
    
    sEMA__sSMA = MA_buysell_indicators(hybridMA_pairs[0], df_daily).head(50)
    sSMA__mEMA = MA_buysell_indicators(hybridMA_pairs[1], df_daily).head(50)
    sEMA__mSMA = MA_buysell_indicators(hybridMA_pairs[2], df_daily).head(50)
    mEMA__mSMA = MA_buysell_indicators(hybridMA_pairs[3], df_daily).head(50)
    
    sEMA__sEMA = MA_buysell_indicators(EMA_pairs[0], df_daily).head(50)
    sEMA__mEMA = MA_buysell_indicators(EMA_pairs[1], df_daily).head(50)
    mEMA__mEMA = MA_buysell_indicators(EMA_pairs[2], df_daily).head(50)
    
    moving_average_pairs = list(sSMA__sSMA, sSMA__mSMA, mSMA__mSMA, sEMA__sSMA, sSMA__mEMA,
         sEMA__mSMA, mEMA__mSMA, sEMA__sEMA, sEMA__mEMA, mEMA__mEMA)

    MA_Profit_Summary = pd.DataFrame({'pair_id':[],
                                   'highest_profit':[],
                                   'mean_profit':[],
                                   'std_dev':[],
                                   'n_std_frm_max':[]}).sort_values(by = 'highest_profit', ascending=False)
    
    for ma_pair in moving_average_pairs:
        for k in MA_Profit_Summary.keys():
            MA_Profit_Summary[k].append(MA_profit_stats(ma_pair)[k])
            
    return MA_Profit_Summary
            
    
SMA_daily = [5, 10, 20, 50, 100, 200] #in days
SMA_hourly = [6.5, 32.5, 65, 130, 390, 1644.5]  #in hours (multiply by 60 mins)
SMA_10m = [60, 120, 390, 780, 1950, 3900, 5850] #in minutes

#sample plots to visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Closing Prices', fontsize=18)
# plt.plot(df_daily['close'], alpha=0.5, label='close')
# plt.plot(df_daily['SMA20'], alpha=0.5, label='SMA20')
# plt.plot(df_daily['SMA50'], alpha=0.5, label='SMA50')
# plt.scatter(df_daily.index, df_daily['buy'], alpha=1, label = 'BUY', marker='o', color='green')
# plt.scatter(df_daily.index, df_daily['sell'], alpha=1, label = 'SELL', marker='o', color='red')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.show()

# print(df_daily[df_daily['signal'] == 1])