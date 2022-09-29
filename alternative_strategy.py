import pandas as pd
import os
from datetime import datetime
import matplotlib as mp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import math
import requests
from scipy import stats
import sklearn
import statistics

class Predictor:
    def __init__(self):
        self.supportData = []
        self.resistanceData = []
        # modelData:
        # (hour, open, close, high, low, volume, TP, MA-TP, std, BOLU, BOLD, Up Move, Down Move, RSI, %K, MACD, SIG, support, resistance, N1, N2, N3, N4)
        
        #Binomial trees for support and resistance. 7 Periods, root holds expected value.
        
        self.nPeriods = 7
        
        self.supportTree = self.Tree(7)
        
        self.resistanceTree = self.Tree(7)
      
    
    class Node:
        def __init__(self):
            self.returns = []
            self.expectedReturn = 0
            self.up = None
            self.down = None
            self.probUp = 1
            self.probDown = 0
        def updateUp(self,val):
            self.up.returns.append(val)
            self.up.expectedReturn = np.average(self.up.returns)
            self.probUp = len(self.up.returns)/(len(self.up.returns) + len(self.down.returns))
            self.probDown = 1 - self.probUp
        def updateDown(self,val):
            self.up.returns.append(val)
            self.up.expectedReturn = np.average(self.up.returns)
            self.probUp = len(self.up.returns)/(len(self.up.returns) + len(self.down.returns))
            self.probDown = 1 - self.probUp
                                                
    class Tree:
        def __init__(self, periods):
            tree = []
            numNodes = 2**periods - 1
            i = 0
            while i < numNodes:
                node = Predictor.Node()
                tree.append(node)
                i += 1
            j =  0
            while j < periods:
                tree[j].up = tree[j*2+1]
                tree[j].down = tree[j*2+2]
                j += 1
    
    def getTrees(self):
        return self.supportTree, self.resistanceTree
    
    def updateSupportTree(self, supportTree):
        for i in self.supportData:
            
            n = [1,3,7,15,31,63,127]
            
            #if price return is negative, update n[i] by 
            
            j = -7
            while j > 0:
                
                if i[j] >= 0:
                    supportTree[n[0]].update(i[j])
                else:
                    for j in range(j+7,7):
                        n[j] += 2**(j-1)
                    supportTree[n[0]].update(i[j])
                
                j += 1
            
        
    def updateResistanceTree(self):
        print()
        
        
    def processSupportAndResistance(self, df, chosen_timeframe):
        
        #adds in support and resistance values for each row of the data frame
        #Range calucated as closing prices over the previous chosen timeframe.
        #Support and resistance calculated as the median of the bottom and top deciles of the previous chosen timeframe's prices, respectively.
        #Range, as well as support and resistance, recalculated on a predefined interval.
        
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
            
            #if currentRow%(chosen_timeframe/rangeRecalculate) == 0:
               # price_range = sorted(list(df['close'][currentRow-chosen_timeframe:currentRow]))
            #    decile = int(len(price_range)/10)
           #     support = np.average(price_range[0:decile])
           #     resistance = np.average(price_range[-decile:-1])
           # df['Support'][currentRow] = support
           # df['Resistance'][currentRow] = resistance
            
            currentRow += 1
            
    def getData(self):
        #Return list of all data
        return self.supportData, self.resistanceData
    
    def MACD(self, df, chosen_timeframe):
        
        #Not updating prev_ema, won't work properly in current form
        
        multiplier_1 = 2/(chosen_timeframe/2+1)
        multiplier_2 = 2/(chosen_timeframe+1)
        prev_ema1 = 0
        prev_ema2 = 0
        df["EMA1"] = (df["close"] * multiplier_12days) + (prev_ema12 * (1 - multiplier_1days))
        df["EMA2"] = (df["close"] * multiplier_26days) + (prev_ema2 * (1 - multiplier_2days))
        df["MACD"] = df["EMA1"] - df["EMA2"]
        df["SIG"] = np.NaN
        j = 9
        while j <= lastIndex:
            mds = list(df['MACD'][j-9:j])
            df['SIG'][j] = np.average(mds)
            j += 1
            
    def processStochasticIndicator(self, df, chosen_timeframe):
        df['n_high'] = df['high'].rolling(chosen_timeframe).max()
        df['n_low'] = df['low'].rolling(chosen_timeframe).min()
        df['%K'] = (df['close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])

    
    def processRSI(self, df, chosen_timeframe):
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
    
    def basicData(self, df, chosen_timeframe):
        df['TP'] = (df['close'] + df['low'] + df['high']) / 3
        df['MA-TP'] = df['TP'].rolling(chosen_timeframe).mean()
        df['std'] = df['TP'].rolling(chosen_timeframe).std()
        df['%std'] = df['std']/df['close']
        df['BOLU'] = df['MA-TP'] + 2 * df['std']
        df['BOLD'] = df['MA-TP'] - 2 * df['std']
    
    def addModelData(self, df, chosen_timeframe):
        
        #Search through dataframe for rows where closing price hits support or resistance. Add that row to model data.
        #hour, openPrice, closePrice, high, low, volume, TP, MA-TP, std, BOLU, BOLD, Up Move, Down Move, RSI, %K, MACD, SIG, support, resistance, N1, N2, N3, N4
        
        currentRow = chosen_timeframe
        
        lastIndex = df.last_valid_index()
        
        while currentRow <= lastIndex:
            
            if df['close'][currentRow] >= df['Support'][currentRow]*0.99 and df['close'][currentRow] <= df['Support'][currentRow]*1.01:
                
                hour = int(str(df['date'][currentRow]).split(' ')[1].split(':')[0])
                openPrice = float(df['open'][currentRow])
                closePrice = float(df['close'][currentRow])
                high = float(df['high'][currentRow])
                low = float(df['low'][currentRow])
                volume = int(df['volume'][currentRow])
                TP = float(df['TP'][currentRow])
                MATP = float(df['MA-TP'][currentRow])
                std = float(df['%std'][currentRow])
                BOLU = float(df['BOLU'][currentRow])
                BOLD = float(df['BOLD'][currentRow])
                upMove = float(df['Up Move'][currentRow])
                downMove = float(df['Down Move'][currentRow])
                RSI = float(df['RSI'][currentRow])
                stochastic = float(df['%K'][currentRow])
                #MACD
                #SIG
                support = float(df['Support'][currentRow])
                resistance = float(df['Resistance'][currentRow])
                halfHour = float(df['close'][currentRow])/float(df['close'][currentRow+2]) - 1
                oneHour = float(df['close'][currentRow])/float(df['close'][currentRow+4]) - 1
                hourHalf = float(df['close'][currentRow])/float(df['close'][currentRow+6]) - 1
                twoHour = float(df['close'][currentRow])/float(df['close'][currentRow+8]) - 1
                threeHour = float(df['close'][currentRow])/float(df['close'][currentRow+12]) - 1
                fourHour = float(df['close'][currentRow])/float(df['close'][currentRow+16]) - 1
                fiveHour = float(df['close'][currentRow])/float(df['close'][currentRow+20]) - 1
                
                self.supportData.append((hour, openPrice, closePrice, high, low, volume, TP, MATP, std, BOLU, BOLD, upMove, downMove, RSI, stochastic, halfHour, oneHour, hourHalf, twoHour, threeHour, fourHour, fiveHour))
            
            elif df['close'][currentRow] >= df['Resistance'][currentRow]*0.99 and df['close'][currentRow] <= df['Resistance'][currentRow]*1.01:
                
                hour = int(str(df['date'][currentRow]).split(' ')[1].split(':')[0])
                openPrice = float(df['open'][currentRow])
                closePrice = float(df['close'][currentRow])
                high = float(df['high'][currentRow])
                low = float(df['low'][currentRow])
                volume = int(df['volume'][currentRow])
                TP = float(df['TP'][currentRow])
                MATP = float(df['MA-TP'][currentRow])
                std = float(df['%std'][currentRow])
                BOLU = float(df['BOLU'][currentRow])
                BOLD = float(df['BOLD'][currentRow])
                upMove = float(df['Up Move'][currentRow])
                downMove = float(df['Down Move'][currentRow])
                RSI = float(df['RSI'][currentRow])
                stochastic = float(df['%K'][currentRow])
                #MACD
                #SIG
                support = float(df['Support'][currentRow])
                resistance = float(df['Resistance'][currentRow])
                halfHour = float(df['close'][currentRow+2])/float(df['close'][currentRow]) - 1
                oneHour = float(df['close'][currentRow+4])/float(df['close'][currentRow]) - 1
                hourHalf = float(df['close'][currentRow+6])/float(df['close'][currentRow]) - 1
                twoHour = float(df['close'][currentRow+8])/float(df['close'][currentRow]) - 1
                threeHour = float(df['close'][currentRow+12])/float(df['close'][currentRow]) - 1
                fourHour = float(df['close'][currentRow+16])/float(df['close'][currentRow]) - 1
                fiveHour = float(df['close'][currentRow+20])/float(df['close'][currentRow]) - 1
                
                self.resistanceData.append((hour, openPrice, closePrice, high, low, volume, TP, MATP, std, BOLU, BOLD, upMove, downMove, RSI, stochastic, halfHour, oneHour, hourHalf, twoHour, threeHour, fourHour, fiveHour))
            
            currentRow += 1
    
    def loadData(self, dataframe, chosen_timeframe):
        
        
        #Preprocess data for each row, add it into pastData
        
        print('Loading dataframe: ', dataframe)
        df = pd.read_csv(dataframe)
        
        
        print('Processing basic data...')
        self.basicData(df, chosen_timeframe)
        
        print('Processing RSI...')
        self.processRSI(df, chosen_timeframe)
        
        print('Processing support and resistance levels...')
        self.processSupportAndResistance(df, chosen_timeframe)
        
        print('Processing stochastic indicator values...')
        self.processStochasticIndicator(df, chosen_timeframe)
        
        print('Completed loading ', dataframe)
        
        print("Adding dataframe data to model...")
        self.addModelData(df, chosen_timeframe)
        
        print("Updating support binomial tree...")
        #self.updateSupportTree(self.supportTree)
        
        print("Updating resistance binomial tree...")
        #self.updateResistanceTree()
        
        print("Completed loading.")
        
        
    def predictMove(self, openPrice, closePrice, high, low, rsi, rsiChange, rsi3point, stoch, bolUpper, bolLower, std, maDif):
        print('')
        