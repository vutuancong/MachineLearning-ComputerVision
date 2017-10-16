Python 3.6.0 (v3.6.0:41df79263a11, Dec 23 2016, 07:18:10) [MSC v.1900 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> 
import pandas as pd
import Quandl

df = Quandl.get('WIKI/GOOGL')

df = df[['Adj.Open','Adj.High','Adj.Low','Adj.Close','Adj.Volume']]
df['HL_PCT'] = (df['Adj.High'] - df['Adj.Close']) / df['Adj.Close'] *100.0
df['PCT_change'] = (df['Adj.Close'] - df['Adj.Open']) / df['Adj.Open'] *100.0

df = df[['Adj.Close','HL_PCT','PCT_change','Adj.Volume']]

print (df.head())
