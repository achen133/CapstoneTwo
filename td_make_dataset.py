from config import ACCT_NUMBER, API_KEY, CALLBACK_URL, JSON_PATH
from td.client import TDClient
import pprint
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ta.volatility import DonchianChannel

#connecting to tdameritrade w/ credentials
td_client = TDClient(client_id = API_KEY, redirect_uri = CALLBACK_URL, account_number = ACCT_NUMBER, credentials_path = JSON_PATH)
td_client.login()

#establish parameters to pull stock data in a specified range from start date to present
def datetime_to_ms(date):
    return str(int(round(date.timestamp() * 1000)))

# datetime(year, month, day, hour, minute, second, microsecond)
howfarback = 365
# present_date = datetime.now()
# past_date = datetime.now() - timedelta(days = howfarback)
present_date = datetime(2017, 12, 31, 23, 59, 59, 999999)
past_date = datetime(2017, 1, 1, 00, 00, 000000)
present_date = str(int(round(present_date.timestamp() * 1000))) #convert datetime to ms
past_date = str(int(round(past_date.timestamp() * 1000)))

#formulate request
daily_data = td_client.get_price_history(symbol = 'NFLX', 
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

#function takes the buy&sell list with extra vals as '0' and outputs the profit 
def generate_returns(buy_lst, sell_lst, signal_lst, n_shares):
    
    buy = [list([i, n, signal_lst[i]]) for i, n in enumerate(buy_lst) if n > 0]
    sell = [list([i, n, signal_lst[i]]) for i, n in enumerate(sell_lst) if n > 0]
    trades = buy + sell
    trades.sort()
    
    trade_index, trade_price = ([] for lst in range(2))
    for trade in trades:
        trade_index.append(trade[0])
        trade_price.append(round(-1*trade[1]*trade[2]*n_shares, 2))
    profit = round(sum(trade_price), 2)
    
    return trade_index, trade_price, profit

#function takes an indicator profit summary & profit col name and outputs a subset of df's max value(s)
def get_top_profits(profit_summary, profit_col):
    _profits = []
    if profit_summary[profit_col].max() > 0:
        top_profits = profit_summary[profit_summary[profit_col] == profit_summary[profit_col].max()]
        _profits.append(top_profits)
    elif profit_summary[profit_col].max() == 0:
        print('[ERROR] Zero Profit')
        top_profits = pd.DataFrame(index=profit_summary.index, columns=profit_summary.keys())
        _profits.append(top_profits)
    elif profit_summary[profit_col].max() < 0:
        print('[ERROR] Negative Profit: ' + str(profit_summary[profit_col].max()))
        top_profits = pd.DataFrame(index=profit_summary.index, columns=profit_summary.keys())
        _profits.append(top_profits)
    df_profits = pd.concat(_profits, ignore_index=True)
    return df_profits
    
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

#list of all possible combinations for SMA vs EMA; s->short & m->mid
SMA_pairs = ['sSMA__sSMA', 'sSMA__mSMA', 'mSMA__mSMA']
hybridMA_pairs = ['sEMA__sSMA', 'sSMA__mEMA', 'sEMA__mSMA', 'mEMA__mSMA']
EMA_pairs = ['sEMA__sEMA', 'sEMA__mEMA', 'mEMA__mEMA']
MA_all_pairs = SMA_pairs + hybridMA_pairs + EMA_pairs

def MA_identify(pair):
    return MA_all_pairs.index(pair)

def MA_strategy2(price, MA_pair1, MA_pair2):
    
    signal_ma, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if MA_pair1[i] >= MA_pair2[i]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_ma.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_ma.append(0)
        elif MA_pair1[i] <= MA_pair2[i]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_ma.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_ma.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_ma.append(0)
    
    return buy, sell, signal_ma

#function to return an ordered list in a dict of the most profitable MA strategy given the input: MA_pair
def MA_strategy(MA_pair, data):
    
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
    p1_list, p2_list, r_profits, MA_trade_index, buysell_prices, buysell_dates = ([] for lst in range(6))
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
            MA_trade_index.append(trade_index)
            
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
                               'MA_id': MA_identify(MA_pair),
                               'p_MA1': p1_list,
                               'p_MA2': p2_list,
                               'trade_index': MA_trade_index,
                               'buy/sell': buysell_prices,
                               'profit': r_profits}).sort_values(by = 'profit', ascending=False)

    return MA_pair_df
    
def MA_parameters(MA_pair, data):
    
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
    p1_list, p2_list, trade_index, trade_price, profits = ([] for lst in range(5))
    for p1 in MA1:
        for p2 in MA2:
            df = data.copy()
            name_MA1 = pair[0] + str(p1)
            name_MA2 = pair[1] + str(p2)
            df[name_MA1] = hybrid_MA(MA_func, df, p1)
            df[name_MA2] = hybrid_MA(MA_func, df, p2)
            
            MA_buy_prices, MA_sell_prices, signal_ma = MA_strategy2(df['close'], df[name_MA1], df[name_MA2])
            MA_trade_index, MA_trade_price, MA_profit = generate_returns(MA_buy_prices, MA_sell_prices, signal_ma, 1)

            p1_list.append(p1)
            p2_list.append(p2)
            trade_index.append(MA_trade_index)
            trade_price.append(MA_trade_price)
            profits.append(MA_profit)

    MA_profits_summary = pd.DataFrame({'MA_id': MA_identify(MA_pair),
                                       'p_MA1': p1_list,
                                       'p_MA2': p2_list,
                                       'MA_trade_index': trade_index,
                                       'MA_trade_price': trade_price,
                                       'MA_profits': profits}).sort_values(by='MA_profits', ascending=False).reset_index()
    
    MA_winners = get_top_profits(MA_profits_summary, 'MA_profits')
    print(type(MA_winners))
    return MA_winners

def MA_profit_stats(dataframe):
    profit_details = pd.DataFrame({'MA_id':dataframe['MA_id'].values,
                                   'MA_pair1':dataframe['p_MA1'].values,
                                   'MA_pair2':dataframe['p_MA2'].values,
                                   'highest_profit':dataframe['profit'].max(),
                                   'mean_profit':dataframe['profit'].mean(),
                                   'std_dev':dataframe['profit'].std(ddof=0)}).reset_index() #number of standard deviations b/e the mean and the highest profit
    
    return profit_details

def run_MA_func(stock_data, n_samples):
    df = stock_data.copy()
    
    sSMA__sSMA = MA_strategy(SMA_pairs[0], df).head(n_samples)
    sSMA__mSMA = MA_strategy(SMA_pairs[1], df).head(n_samples)
    mSMA__mSMA = MA_strategy(SMA_pairs[2], df).head(n_samples)
    
    sEMA__sSMA = MA_strategy(hybridMA_pairs[0], df).head(n_samples)
    sSMA__mEMA = MA_strategy(hybridMA_pairs[1], df).head(n_samples)
    sEMA__mSMA = MA_strategy(hybridMA_pairs[2], df).head(n_samples)
    mEMA__mSMA = MA_strategy(hybridMA_pairs[3], df).head(n_samples)
    
    sEMA__sEMA = MA_strategy(EMA_pairs[0], df).head(n_samples)
    sEMA__mEMA = MA_strategy(EMA_pairs[1], df).head(n_samples)
    mEMA__mEMA = MA_strategy(EMA_pairs[2], df).head(n_samples)
    
    moving_average_pairs = [sSMA__sSMA, sSMA__mSMA, mSMA__mSMA, sEMA__sSMA, sSMA__mEMA,
                            sEMA__mSMA, mEMA__mSMA, sEMA__sEMA, sEMA__mEMA, mEMA__mEMA]
    
    dd = defaultdict(list)
    for MApair in moving_average_pairs:
        pair_stats = MA_profit_stats(MApair).iloc[0]
        for k, v in pair_stats.items():
            dd[k].append(v)
            
    MA_Profit_Summary = pd.DataFrame(dd).sort_values(by = 'highest_profit', axis=0, ascending=False).reset_index()
    
    MA_winner_index = MA_Profit_Summary[MA_Profit_Summary['highest_profit'] == MA_Profit_Summary['highest_profit'].max()].index
    MA_winners = defaultdict(list)
    for i in MA_winner_index:
        MA_winner = moving_average_pairs[int(MA_Profit_Summary['MA_id'][i])].iloc[int(MA_Profit_Summary['index'][i]), :]
        for k, v in MA_winner.items():
            MA_winners[k].append(v)
    MA_winners = pd.DataFrame(MA_winners)
    
    return MA_Profit_Summary.head(10), MA_winners.T

def run_MA_func2(stock_data):
    df = stock_data.copy()
    
    sSMA__sSMA = MA_parameters(SMA_pairs[0], df)
    sSMA__mSMA = MA_parameters(SMA_pairs[1], df)
    mSMA__mSMA = MA_parameters(SMA_pairs[2], df)
    
    sEMA__sSMA = MA_parameters(hybridMA_pairs[0], df)
    sSMA__mEMA = MA_parameters(hybridMA_pairs[1], df)
    sEMA__mSMA = MA_parameters(hybridMA_pairs[2], df)
    mEMA__mSMA = MA_parameters(hybridMA_pairs[3], df)
    
    sEMA__sEMA = MA_parameters(EMA_pairs[0], df)
    sEMA__mEMA = MA_parameters(EMA_pairs[1], df)
    mEMA__mEMA = MA_parameters(EMA_pairs[2], df)
    
    moving_average_pairs = [sSMA__sSMA, sSMA__mSMA, mSMA__mSMA, sEMA__sSMA, sSMA__mEMA,
                            sEMA__mSMA, mEMA__mSMA, sEMA__sEMA, sEMA__mEMA, mEMA__mEMA]
        
    cmoving_average_pairs = pd.concat(moving_average_pairs, ignore_index=True).dropna()
    cmoving_average_pairs = cmoving_average_pairs.sort_values(by = 'MA_profits', axis=0, ascending=False).reset_index()
    print(cmoving_average_pairs)
    # dd = defaultdict(list)
    # for MApair in moving_average_pairs:
    #     print(MApair)
    #     for k, v in MApair.items():
    #         dd[k].append(v)
            
    # MA_Profit_Summary = pd.DataFrame(dd).sort_values(by = 'MA_profits', axis=0, ascending=False).reset_index()
    
    MA_winner_index = get_top_profits(cmoving_average_pairs, 'MA_profits')
    
    # MA_winner_index = MA_Profit_Summary[MA_Profit_Summary['MA_profits'] == MA_Profit_Summary['MA_profits'].max()].index
    # MA_winners = defaultdict(list)
    # for i in MA_winner_index:
    #     MA_winner = moving_average_pairs[int(MA_Profit_Summary['MA_id'][i])].iloc[int(MA_Profit_Summary['index'][i]), :]
    #     for k, v in MA_winner.items():
    #         MA_winners[k].append(v)
    # MA_winners = pd.DataFrame(MA_winners)
    return MA_winner_index
    # return MA_Profit_Summary.head(10), MA_winners

def MACD_strategy(price, MACD, signal_line):
    signal_macd, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if MACD[i] > signal_line[i]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_macd.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_macd.append(0)
        elif MACD[i] < signal_line[i]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_macd.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_macd.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_macd.append(0)
    
    return buy, sell, signal_macd

def MACD_parameters(data):
    df = data.copy()
    MACD_profits_summary = pd.DataFrame({})
    
    df['EMA12'] = EMA(df, 12)
    df['EMA26'] = EMA(df, 26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['signal_line'] = EMA(df, 9)
    df['hist'] = df['MACD'] - df['signal_line']
    df['rolling_hist_std'] = df['close'].rolling(window=12).std()
    
    MACD_buy_prices, MACD_sell_prices, signal_macd = MACD_strategy(df['close'], df['MACD'], df['signal_line'])
    MACD_trade_index, MACD_trade_price, MACD_profit = generate_returns(MACD_buy_prices, MACD_sell_prices, signal_macd, 1)
    df['MACD_profits'] = MACD_profit
    
    dd = defaultdict(list)
    for k, v in df.items():
        dd[k].extend(v)
       
    MACD_profit_entry = pd.DataFrame(dd)
    MACD_profits_summary = pd.concat([MACD_profits_summary, MACD_profit_entry])
    MACD_profits_summary = MACD_profits_summary.sort_values(by = 'MACD_profits', axis=0, ascending=False)
              
    return MACD_trade_index, MACD_trade_price, MACD_profit

# print(MACD_parameters(df_daily))
print('------------------------------------------------')

#squeeze indicators: Bollinger Bands, Keltner Channels, ATR, TTM Squeeze
df_squeeze = df_daily.copy()
squeeze_SMA= list(range(4, 43))
squeeze_EMA = list(range(4, 43))
squeeze_period = list(range(4, 43))
std_dev_multiplier = list(np.arange(1.0, 2.5, 0.1))
ATR_multiplier = list(np.arange(1.0, 2.5, 0.1))

def BB_strategy(price, lower_bb, upper_bb):
    
    signal_bb, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if price[i] < lower_bb[i] and price[i-1] > lower_bb[i-1]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_bb.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_bb.append(0)
        elif price[i] > upper_bb[i] and price[i-1] < upper_bb[i-1]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_bb.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_bb.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_bb.append(0)
    
    return buy, sell, signal_bb

def KC_strategy(price, lower_kc, upper_kc):
    
    signal_kc, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(0, len(price)-1):
        if price[i] < lower_kc[i] and price[i+1] > price[i]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_kc.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_kc.append(0)
        elif price[i] > upper_kc[i] and price[i+1] < price[i]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_kc.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_kc.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_kc.append(0)
    
    return buy, sell, signal_kc

def squeeze_strategy(price, lower_bb, upper_bb, lower_kc, upper_kc):

    signal_squeeze, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if lower_kc[i] > lower_bb[i] and upper_kc[i] > upper_bb[i]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_squeeze.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_squeeze.append(0)
        elif lower_kc[i] < lower_bb[i] and upper_kc[i] > upper_bb[i]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_squeeze.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_squeeze.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_squeeze.append(0)
    
    return buy, sell, signal_squeeze

def ttm_squeeze_strategy(price, in_squeeze, DC_mean):
    
    signal_ttm, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        try:
            if in_squeeze[i] != 1 and in_squeeze[i-1] == 1 and DC_mean[i] > 0:
                if signal != 1:
                    buy.append(price[i])
                    sell.append(0)
                    signal = 1
                    signal_ttm.append(signal)
                else:
                    buy.append(0)
                    sell.append(0)
                    signal_ttm.append(0)
            if in_squeeze[i] != 1 and in_squeeze[i-1] == 1 and DC_mean[i] < 0:
                if signal != -1:
                    buy.append(0)
                    sell.append(price[i])
                    signal = -1
                    signal_ttm.append(signal)
                else:
                    buy.append(0)
                    sell.append(0)
                    signal_ttm.append(0)
            else:
                buy.append(0)
                sell.append(0)
                signal_ttm.append(0)
        except KeyError:
            pass
    
    return buy, sell, signal_ttm

def BB_parameters(data):
    period, multiplier, trade_index, trade_price, profits = ([] for lst in range(5))
    for p1 in squeeze_SMA:
        for p2 in std_dev_multiplier:
            df = data.copy()
            df['SMA'] = SMA(df, p1)
            df['std_dev'] = df['close'].rolling(window=p1).std()
            df['lower_bb'] = df['SMA'] - (p2 * df['std_dev'])
            df['upper_bb'] = df['SMA'] + (p2 * df['std_dev'])
            
            BB_buy_prices, BB_sell_prices, signal_bb = BB_strategy(df['close'], df['lower_bb'], df['upper_bb'])
            BB_trade_index, BB_trade_price, BB_profit = generate_returns(BB_buy_prices, BB_sell_prices, signal_bb, 1)

            period.append(p1)
            multiplier.append(p2)
            trade_index.append(BB_trade_index)
            trade_price.append(BB_trade_price)
            profits.append(BB_profit)
            
    BB_profits_summary = pd.DataFrame({'SMA_period': period,
                                       'std_dev_multiplier': multiplier,
                                       'BB_trade_index': trade_index,
                                       'BB_trade_price': trade_price,
                                       'BB_profits': profits}).sort_values(by='BB_profits', ascending=False).reset_index()
    
    BB_winners = get_top_profits(BB_profits_summary, 'BB_profits')
    
    return BB_winners

def KC_parameters(data):
    period, multiplier, trade_index, trade_price, profits = ([] for lst in range(5))
    for p1 in squeeze_EMA:
        for p2 in ATR_multiplier:
            df = data.copy()
            df['EMA'] = EMA(df, p1)
            df['TR'] = abs(df['high'] - df['low'])
            df['ATR'] = df['TR'].rolling(window=p1).mean()

            #calculating lower and upper Keltner Channels with EMA20 & ATR
            df['lower_kc'] = df['EMA'] - (df['ATR'] * p2)
            df['upper_kc'] = df['EMA'] + (df['ATR'] * p2)
            
            KC_buy_prices, KC_sell_prices, signal_kc = KC_strategy(df['close'], df['lower_kc'], df['upper_kc'])
            KC_trade_index, KC_trade_price, KC_profit = generate_returns(KC_buy_prices, KC_sell_prices, signal_kc, 1)

            period.append(p1)
            multiplier.append(p2)
            trade_index.append(KC_trade_index)
            trade_price.append(KC_trade_price)
            profits.append(KC_profit)
            
    KC_profits_summary = pd.DataFrame({'EMA_period': period,
                                       'ATR_multiplier': multiplier,
                                       'KC_trade_index': trade_index,
                                       'KC_trade_price': trade_price,
                                       'KC_profits': profits}).sort_values(by='KC_profits', ascending=False).reset_index()
    
    KC_winners = get_top_profits(KC_profits_summary, 'KC_profits')
    
    return KC_winners

def squeeze_parameters(data):
    period, BB_multiplier, KC_multiplier, trade_index, trade_price, profits = ([] for lst in range(6))
    for p1 in squeeze_period:
        for p2 in std_dev_multiplier:
            for p3 in ATR_multiplier:
                df = data.copy()
                df['SMA'] = SMA(df, p1)
                df['EMA'] = EMA(df, p1)
                df['std_dev'] = df['close'].rolling(window=p1).std()
                df['lower_bb'] = df['SMA'] - (p2 * df['std_dev'])
                df['upper_bb'] = df['SMA'] + (p2 * df['std_dev'])

                df['TR'] = abs(df['high'] - df['low'])
                df['ATR'] = df['TR'].rolling(window=p1).mean()

                #calculating lower and upper Keltner Channels with EMA20 & ATR
                df['lower_kc'] = df['EMA'] - (df['ATR'] * p3)
                df['upper_kc'] = df['EMA'] + (df['ATR'] * p3)
                    
                squeeze_buy_prices, squeeze_sell_prices, signal_squeeze = squeeze_strategy(df['close'], df['lower_bb'], df['upper_bb'], df['lower_kc'], df['upper_kc'])
                squeeze_trade_index, squeeze_trade_price, squeeze_profit = generate_returns(squeeze_buy_prices, squeeze_sell_prices, signal_squeeze, 1)
                
                period.append(p1)
                BB_multiplier.append(p2)
                KC_multiplier.append(p3)
                trade_index.append(squeeze_trade_index)
                trade_price.append(squeeze_trade_price)
                profits.append(squeeze_profit)
                
    squeeze_profits_summary = pd.DataFrame({'squeeze_period': period,
                                            'std_dev_multiplier': BB_multiplier, 
                                            'ATR_multiplier': KC_multiplier,
                                            'squeeze_trade_index': trade_index,
                                            'squeeze_trade_price': trade_price,
                                            'squeeze_profits': profits}).sort_values(by='squeeze_profits', ascending=False).reset_index()
    
    squeeze_winners = get_top_profits(squeeze_profits_summary, 'squeeze_profits')
    
    return squeeze_winners

def ttm_squeeze_parameters(data):
    period, BB_multiplier, KC_multiplier, trade_index, trade_price, profits = ([] for lst in range(6))
    for p1 in squeeze_period:
        for p2 in std_dev_multiplier:
            for p3 in ATR_multiplier:
                df = data.copy()
                df['SMA'] = SMA(df, p1)
                df['EMA'] = EMA(df, p1)
                df['std_dev'] = df['close'].rolling(window=p1).std()
                df['lower_bb'] = df['SMA'] - (p2 * df['std_dev'])
                df['upper_bb'] = df['SMA'] + (p2 * df['std_dev'])

                df['TR'] = abs(df['high'] - df['low'])
                df['ATR'] = df['TR'].rolling(window=p1).mean()

                #calculating lower and upper Keltner Channels with EMA20 & ATR
                df['lower_kc'] = df['EMA'] - (df['ATR'] * p3)
                df['upper_kc'] = df['EMA'] + (df['ATR'] * p3)
                
                #calculating highest high and lowest low across a set period
                DC_volitility = DonchianChannel(df['high'], df['low'], df['close'], p1)
                upper = DC_volitility.donchian_channel_hband()
                lower = DC_volitility.donchian_channel_lband()
                DC_mean = DC_volitility.donchian_channel_mband()
                DC_diff = (df['close'].rolling(window=p1).mean()) - DC_mean
                df['in_squeeze'] = np.where((df['upper_bb'] < df['upper_kc']) & (df['lower_bb'] > df['lower_kc']), 1, 0)
                    
                ttm_buy_prices, ttm_sell_prices, signal_ttm = ttm_squeeze_strategy(df['close'], df['in_squeeze'], DC_mean)
                ttm_trade_index, ttm_trade_price, ttm_profit = generate_returns(ttm_buy_prices, ttm_sell_prices, signal_ttm, 1)
                
                period.append(p1)
                BB_multiplier.append(p2)
                KC_multiplier.append(p3)
                trade_index.append(ttm_trade_index)
                trade_price.append(ttm_trade_price)
                profits.append(ttm_profit)
                
    ttm_profits_summary = pd.DataFrame({'squeeze_period': period,
                                        'std_dev_multiplier': BB_multiplier, 
                                        'ATR_multiplier': KC_multiplier,
                                        'ttm_trade_index': trade_index,
                                        'ttm_trade_price': trade_price,
                                        'ttm_profits': profits}).sort_values(by='ttm_profits', ascending=False).reset_index()
    
    ttm_winners = get_top_profits(ttm_profits_summary, 'ttm_profits')
    
    return ttm_winners

def run_squeeze_func():
    n_samples = 10
    BB_stats = BB_parameters(df_squeeze).head(n_samples)
    BB_mean = BB_stats.iloc[:, -1:].mean()
    BB_std = BB_stats.iloc[:, -1:].std(ddof=0)
    print(BB_stats.iloc[0, :], BB_mean, BB_std)
    
    KC_stats = KC_parameters(df_squeeze).head(n_samples)
    KC_mean = KC_stats.iloc[:, -1:].mean()
    KC_std = KC_stats.iloc[:, -1:].std(ddof=0)
    print(KC_stats.iloc[0, :], KC_mean, KC_std)
    
    squeeze_stats = squeeze_parameters(df_squeeze).head(n_samples)
    squeeze_mean = squeeze_stats.iloc[:, -1:].mean()
    squeeze_std = squeeze_stats.iloc[:, -1:].std(ddof=0)
    print(squeeze_stats.iloc[0, :], squeeze_mean, squeeze_std)

    ttm_squeeze_stats = ttm_squeeze_parameters(df_squeeze).head(n_samples)
    ttm_squeeze_mean = ttm_squeeze_stats.iloc[:, -1:].mean()
    ttm_squeeze_std = ttm_squeeze_stats.iloc[:, -1:].std(ddof=0)
    print(ttm_squeeze_stats.iloc[0, :], ttm_squeeze_mean, ttm_squeeze_std)
# run_squeeze_func()
print('------------------------------------------------')

#ROC to KST momentum indicators
df_momentum = df_daily.copy()
ROC_range = list(range(9, 51))
roc1_range = list(range(8, 11))
roc2_range = list(range(12, 16))
roc3_range = list(range(15, 21))
roc4_range = list(range(20, 31))

def ROC_sample(data, period):
    close_diff = data['close'].diff(period)
    previous_period = data['close'].shift(period)
    ROC = (close_diff/previous_period)*100
    return ROC

def ROC_strategy(price, n_roc):
    signal_roc, buy, sell= ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if n_roc[i] > 0 and n_roc[i-1] < 0:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_roc.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_roc.append(0)
        elif n_roc[i] < 0 and n_roc[i-1] > 0:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_roc.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_roc.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_roc.append(0)
    
    return buy, sell, signal_roc

def KST_strategy(price, kst, signal_line):
    signal_kst, buy, sell = ([] for lst in range(3))
    signal = 0
    
    for i in range(len(price)):
        if kst[i] > signal_line[i] and kst[i] < 0 and kst[i-1] < signal_line[i-1]:
            if signal != 1:
                buy.append(price[i])
                sell.append(0)
                signal = 1
                signal_kst.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_kst.append(0)
        elif kst[i] < signal_line[i] and kst[i] > 0 and kst[i-1] > signal_line[i-1]:
            if signal != -1:
                buy.append(0)
                sell.append(price[i])
                signal = -1
                signal_kst.append(signal)
            else:
                buy.append(0)
                sell.append(0)
                signal_kst.append(0)
        else:
            buy.append(0)
            sell.append(0)
            signal_kst.append(0)
    
    return buy, sell, signal_kst

def ROC_parameters(data):
    
    period, trade_index, trade_price, profits = ([] for lst in range(4))
    for n in ROC_range:
        n_roc = ROC_sample(data, n)
        ROC_buy_prices, ROC_sell_prices, signal_roc = ROC_strategy(data['close'], n_roc)
        ROC_trade_index, ROC_trade_price, ROC_profit = generate_returns(ROC_buy_prices, ROC_sell_prices, signal_roc, 1)
        
        period.append(n)
        trade_index.append(ROC_trade_index)
        trade_price.append(ROC_trade_price)
        profits.append(ROC_profit)
            
    ROC_profits_summary = pd.DataFrame({'ROC_period': period,
                                       'ROC_trade_index': trade_index,
                                       'ROC_trade_price': trade_price,
                                       'ROC_profits': profits}).sort_values(by='ROC_profits', ascending=False).reset_index()
        
    return ROC_profits_summary
# print(ROC_parameters(df_momentum))

def KST_parameters(data, sma1, sma2, sma3, sma4, kst_sma=9):
    
    KST_profits_summary = pd.DataFrame({})
    for pr1 in roc1_range:
        for pr2 in roc2_range:
            for pr3 in roc3_range:
                for pr4 in roc4_range:
                    rcma1 = ROC_sample(data, pr1).rolling(sma1).mean()
                    rcma2 = ROC_sample(data, pr2).rolling(sma2).mean()
                    rcma3 = ROC_sample(data, pr3).rolling(sma3).mean()
                    rcma4 = ROC_sample(data, pr4).rolling(sma4).mean()
                    kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
                    signal_line = kst.rolling(kst_sma).mean()
                    
                    KST_buy_prices, KST_sell_prices, signal_KST = KST_strategy(data['close'], kst, signal_line)
                    KST_trade_index, KST_trade_price, KST_profit = generate_returns(KST_buy_prices, KST_sell_prices, signal_KST, 1)
                    
                    key_names = ['sma1', 'sma2', 'sma3', 'sma4', 'roc1', 'roc2', 'roc3', 'roc4', 'kst_sma', 'KST_trade_index', 'KST_trade_price', 'KST_profits']
                    value_names = [sma1, sma2, sma3, sma4, pr1, pr2, pr3, pr4, kst_sma, KST_trade_index, KST_trade_price, KST_profit]
                    
                    dd = defaultdict(list)
                    for k, v in zip(key_names, value_names):
                        for key in k.split(','):
                            dd[key.strip()].append(v)
                            
                    KST_profit_entry = pd.DataFrame(dd)
                    KST_profits_summary = pd.concat([KST_profits_summary, KST_profit_entry])
    KST_profits_summary = KST_profits_summary.sort_values(by = 'KST_profits', axis=0, ascending=False)
    KST_winner = KST_profits_summary.iloc[0, :]
              
    return KST_winner
# print(KST_parameters(df_momentum, 10, 10, 10, 15))

def run_KST_func():
    df_momentum = df_daily.copy()
    kst_sma = 9
    sma1_range = list(range(8, 11))
    sma2_range = list(range(10, 14))
    sma3_range = list(range(10, 16))
    sma4_range = list(range(15, 21))
    
    roc1_range = list(range(8, 11))
    roc2_range = list(range(12, 16))
    roc3_range = list(range(15, 21))
    roc4_range = list(range(20, 31))
    
    sma_vals = [10, 10, 10, 15]
    KST_profits_all = pd.DataFrame({})

    for pr1 in roc1_range:
        for pr2 in roc2_range:
            for pr3 in roc3_range:
                for pr4 in roc4_range:
                    KST_stats = KST_parameters(df_momentum, sma_vals[0], sma_vals[1], sma_vals[2], sma_vals[3], pr1, pr2, pr3, pr4, kst_sma)
                    KST_profits_all = pd.concat([KST_profits_all, KST_stats])
    KST_profits_all = KST_profits_all.sort_values(by = 'KST_profits', axis=0, ascending=False)
                
    return KST_profits_all
# print(run_KST_func())
print('--------------------------------------------------')



def backtest(ticker_symbol, choose_year):
    
    present_date = datetime(choose_year, 12, 31, 23, 59, 59, 999999)
    past_date = datetime(choose_year, 1, 1, 00, 00, 000000)

    raw_stock_data = td_client.get_price_history(symbol = ticker_symbol, 
                                          period_type = 'month',
                                          frequency_type = 'daily',
                                          frequency = 1, 
                                          start_date = datetime_to_ms(past_date),
                                          end_date = datetime_to_ms(present_date), 
                                          extended_hours = True)
    
    df = pd.DataFrame(raw_stock_data['candles'])
    df['date'] = [datetime.fromtimestamp(int(dates)/1000).strftime('%Y-%m-%d') for dates in df['datetime']]
    df = df.set_index(pd.DatetimeIndex(df['date'])).drop(columns = ['datetime', 'date']).reset_index()
    
    n_trials = 5
    # moving_average = run_MA_func(df, n_trials)
    moving_average2 = run_MA_func2(df)
    # bollinger_band = BB_parameters(df)
    # keltner_channel = KC_parameters(df)
    # bbkc_squeeze = squeeze_parameters(df)
    # ttm_squeeze = ttm_squeeze_parameters(df)
    # bollinger_band, keltner_channel, bbkc_squeeze, ttm_squeeze
    return moving_average2
print(backtest('NFLX', 2017))
    


KST_daily = [10,10,10,15,10,15,20,30,9]
KST_weekly = [10,13,15,20,10,13,15,20,9]
KST_monthly = [6,6,6,9,9,12,18,24,9]

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