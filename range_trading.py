import pandas as pd
import time
import multiprocessing as mp
import pandas_ta as ta
import numpy as np

# local imports
from backtester import engine, tester
from backtester import API_Interface as api

training_period = 20  # How far the rolling average takes into calculation
# Number of Standard Deviations from the mean the Bollinger Bands sit
standard_deviations = 3.5

'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call
'''


def logic(account, lookback):  # Logic function to be used for each time interval in backtest

    dayBefore = len(lookback) - 2
    today = len(lookback) - 1

    '''
        Develop Logic Here
    '''

    if (today > training_period):  # If the lookback is long enough to calculate the Bollinger Bands
        mac = lookback['MACD'][today]
        ma = lookback['SIG'][today]

        ##  SELL  ##
        # if lookback['rsi'][today] >= 70:
        if lookback['%K'][today] >= 80:
            # If today's price is above the upper Bollinger Band, enter a short position
            if (lookback['close'][today] > lookback['BOLU'][today]):
                if mac > ma:
                    if (lookback["SIG"][dayBefore] > lookback["MACD"][dayBefore]):
                        if (mac - lookback["MACD"][dayBefore] > 0):
                            for position in account.positions:  # Close all current positions
                                account.close_position(
                                    position, 1, lookback['close'][today])
                            if (account.buying_power > 0):
                                # Enter a short position
                                account.enter_position(
                                    'short', account.buying_power, lookback['close'][today])

        ##  BUY  ##
        # if lookback['rsi'][today] <= 30:
        if lookback['%K'][today] <= 20:
            # If current price is below lower Bollinger Band, enter a long position
            if (lookback['close'][today] < lookback['BOLD'][today]):
                if mac < ma:
                    if (lookback["SIG"][dayBefore] < lookback["MACD"][dayBefore]):
                        if (mac - lookback["MACD"][dayBefore] < 0):
                            for position in account.positions:  # Close all current positions
                                account.close_position(
                                    position, 1, lookback['close'][today])
                            if (account.buying_power > 0):
                                # Enter a long position
                                account.enter_position(
                                    'long', account.buying_power, lookback['close'][today])


'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.

    Input:  list_of_stocks - a list of stock data csvs to be processed

    Output: list_of_stocks_processed - a list of processed stock data csvs
'''


def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []
    multiplier_12days = 2/13
    multiplier_26days = 2/27
    prev_ema12 = 0
    prev_ema26 = 0
    row = 1
    k_period = 14
    prev_close = 0

    for stock in list_of_stocks:

        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])

        '''
        
        Modify Processing of Data To Suit Personal Requirements.
        
        '''
        df['rowNo'] = row

        # for i in range(len(df)):

        #     ### RSI INDICATOR ###
        #     change = df['close'][i] - prev_close

        #     if change > 0:

        #         df['Up Move']= df['change']
        #         df['Down Move'][i] = 0
        #     else:
        #         df['Up Move'][i] = 0
        #         df['Down Move'][i] = df['change'][i] * -1

        #     if i >= 20:
        #         avgUp = df['Up Move'].rolling(20).mean()
        #         avgDown = df['Down Move'].rolling(20).mean()
        #         df['rsi'][i] =  100 - 100/(1+(avgUp/avgDown))
    # r = 1
    # df['Up Move'] = np.NaN
    # df['Down Move'] = np.NaN
    # lastIndex = df.last_valid_index()
    # while r <= lastIndex:
    #     change = df['close'][r] - df['close'][r-1]
    #     if change >= 0:
    #         df['Up Move'][r] = change
    #         df['Down Move'][r] = 0
    #     else:
    #         df['Up Move'][r] = 0
    #         df['Down Move'][r] = change * -1
    #     if r >= 20:
    #         upSum = 0
    #         for j in range(r-20,r+1):
    #             upSum += df['Up Move'][j]
    #         avgUp = upSum/20
    #         downSum = 0
    #         for k in range(r-20,r):
    #             downSum += df['Down Move'][k]
    #         avgDown = downSum/20
    #         rsi = 100 - 100/(1+(avgUp/avgDown))
    #         df['rsi'][r] = rsi

    #     r += 1

        ### MAC D INDICATOR ###
        row += 1
        df['TP'] = (df['close'] + df['low'] + df['high']) / \
            3  # Calculate Typical Price
        # Calculate Moving Average of Typical Price
        df['MA-TP'] = df['TP'].rolling(training_period).mean()
        df["EMA12"] = (df["close"] * multiplier_12days) + \
            (prev_ema12 * (1 - multiplier_12days))  # 12 period EMA
        df["EMA26"] = (df["close"] * multiplier_26days) + \
            (prev_ema26 * (1 - multiplier_26days))  # 26 period EMA
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["SIG"] = df.groupby((df['rowNo'] - 1) // 9)['MACD'].apply(lambda x: (x.shift(1) + x.shift(
            2) + x.shift(3) + x.shift(4) + x.shift(5) + x.shift(6) + x.shift(7) + x.shift(8) + x.shift(9)) / 9)

        ### STOIC INDICATOR ###

        # Adds a "n_high" column with max value of previous 14 periods
        df['n_high'] = df['high'].rolling(k_period).max()
        # Adds an "n_low" column with min value of previous 14 periods
        df['n_low'] = df['low'].rolling(k_period).min()
        # Uses the min/max values to calculate the %k (as a percentage)
        df['%K'] = (df['close'] - df['n_low']) * \
         100 / (df['n_high'] - df['n_low'])

        ## BOLLINGER BANDS ##

        # Calculate Standard Deviation
        df['std'] = df['TP'].rolling(training_period).std()
        df['BOLU'] = df['MA-TP'] + standard_deviations * \
        df['std']  # Calculate Upper Bollinger Band
        df['BOLD'] = df['MA-TP'] - standard_deviations * \
        df['std']  # Calculate Lower Bollinger Band

        df.to_csv("data/" + stock + "_Processed.csv",
                  index=False)  # Save to CSV

        prev_ema12 = df["EMA12"]
        prev_ema26 = df["EMA26"]
        prev_close = df["close"]

        list_of_stocks_processed.append(stock + "_Processed")
        
    return list_of_stocks_processed


if __name__ == "__main__":
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"]
    # List of stock data csv's to be tested, located in "data/" folder
    list_of_stocks = ["TSLA_2020-03-09_2022-01-28_15min", "AAPL_2020-03-24_2022-02-12_15min"]
    list_of_stocks_proccessed = preprocess_data(list_of_stocks)  # Preprocess the data
    # Run backtest on list of stocks using the logic function
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True)

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold", "Strategy", "Longs", "Sells", "Shorts", "Covers", "Stdev_Strategy", "Stdev_Hold", "Stock"])  # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False)  # Save results to csv
