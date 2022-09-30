import pandas as pd
import time
import multiprocessing as mp
import numpy as np

# local imports
from backtester import engine, tester
from backtester import API_Interface as api

pd.options.mode.chained_assignment = None  # default='warn'

training_period = 20 # How far the rolling average takes into calculation
standard_deviations = 3.5 # Number of Standard Deviations from the mean the Bollinger Bands sit

'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call
'''

def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    
    today = len(lookback)-1

    '''
    
    Develop Logic Here
    
    '''

    #Buy at support
    hour = int(str(lookback['date'][today]).split(' ')[1].split(':')[0])
    if hour < 15 and lookback['close'][today] > lookback['BOLD'][today] and lookback['RSI'][today] > 40 and lookback['%K'][today] < 50 and lookback['%std'][today] > 0.015:
        #buy, sell 6 periods later
        for position in account.positions:  # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        if (account.buying_power > 0):
            # Enter a long position
            account.enter_position('long', account.buying_power, lookback['close'][today])



    #Sell at resistance
    if hour > 15 and lookback['close'][today] < lookback['BOLU'][today] and lookback['RSI'][today] < 60 and lookback['%K'][today] < 50 and lookback['%std'][today] > 0.015:
        #sell, buy 6 periods later
        for position in account.positions:  # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        if (account.buying_power > 0):
            # Enter a short position
            account.enter_position('short', account.buying_power, lookback['close'][today])




'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.

    Input:  list_of_stocks - a list of stock data csvs to be processed

    Output: list_of_stocks_processed - a list of processed stock data csvs
'''
def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []
    chosen_timeframe = 64*5
    for stock in list_of_stocks:
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])

        '''
        
        Modify Processing of Data To Suit Personal Requirements.
        
        '''
        #Basic data
        df['TP'] = (df['close'] + df['low'] + df['high']) / 3
        df['MA-TP'] = df['TP'].rolling(chosen_timeframe).mean()
        df['std'] = df['TP'].rolling(chosen_timeframe).std()
        df['%std'] = df['std']/df['close']
        df['BOLU'] = df['MA-TP'] + 2 * df['std']
        df['BOLD'] = df['MA-TP'] - 2 * df['std']

        #Stochastic indicator
        df['n_high'] = df['high'].rolling(chosen_timeframe).max()
        df['n_low'] = df['low'].rolling(chosen_timeframe).min()
        df['%K'] = (df['close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])

        #RSI
        #This part is quite slow, apologies to your PC
        df['Up Move'] = np.NaN
        df['Down Move'] = np.NaN
        df['RSI'] = np.NaN

        lastIndex = df.last_valid_index()
        i = 1
        while i <= lastIndex:
            change = df['close'][i] - df['close'][i-1]
            if change >= 0:
                df['Up Move'][i] = change
                df['Down Move'][i] = 0
            else:
                df['Up Move'][i] = 0
                df['Down Move'][i] = change * -1
            if i >= chosen_timeframe:
                upSum = 0
                downSum = 0
                for j in range(i-chosen_timeframe, i+1):
                    upSum += df['Up Move'][j]
                    downSum += df['Down Move'][j]
                avgUp = upSum/chosen_timeframe
                avgDown = downSum/chosen_timeframe
                rsi = 100 - (100/(1+(avgUp/avgDown)))
                df['RSI'][i] = rsi
    
            i += 1

        #Support and Resistance
        df['Support'] = np.NaN
        df['Resistance'] = np.NaN
        
        price_range = sorted(list(df['close'][0:chosen_timeframe]))
        decile = int(len(price_range)/10)
        support = np.median(price_range[0:decile])
        resistance = np.median(price_range[-decile:-1])
        
        lastIndex = df.last_valid_index()
        currentRow = chosen_timeframe
        currentDay = str(df['date'][chosen_timeframe]).split(' ')[0]
        
        while currentRow < lastIndex:
            #If new day, recalculate range, support, and resistance.
            if str(df['date'][currentRow]).split(' ')[0] != currentDay:
                currentDay = str(df['date'][currentRow]).split(' ')[0]
                price_range = sorted(list(df['close'][currentRow-chosen_timeframe:currentRow]))
                decile = int(len(price_range)/10)
                support = np.average(price_range[0:decile])
                resistance = np.average(price_range[-decile:-1])
            
            df['Support'][currentRow] = support
            df['Resistance'][currentRow] = resistance
            
            currentRow += 1


        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed

if __name__ == "__main__":
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"] 
    list_of_stocks = ["TSLA_2020-03-09_2022-01-28_15min", "AAPL_2020-03-24_2022-02-12_15min"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv