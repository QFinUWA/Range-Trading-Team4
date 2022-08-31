from calendar import c
import pandas as pd
import numpy as np
import matplotlib as mp


df = pd.read_csv('data/AAPL_2020-03-24_2022-02-12_15min_Processed copy.csv')


r = 1
period = 14
lastIndex = df.last_valid_index()
df['Up Move'][0] = 0
df['Down Move'][0] = 0
while r <= lastIndex:
    change = df['close'][r] - df['close'][r-1]
    if change >= 0:
        df['Up Move'][r] = change
        df['Down Move'][r] = 0
    else:
        df['Up Move'][r] = 0
        df['Down Move'][r] = change * -1
    if r >= period:
        upSum = 0
        for j in range(r-period,r+1):
            upSum += df['Up Move'][j]
        avgUp = upSum/period
        #print(avgUp)
        downSum = 0
        for k in range(r-period,r):
            downSum += df['Down Move'][k]
        avgDown = downSum/period
        #print(avgDown)
        #print(avgUp/avgDown)
        rsi = 100 - 100/(1+(avgUp/avgDown))
        df['rsi'][r] = rsi
        #print(rsi)
    r += 1