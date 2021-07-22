import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import sklearn
import pytrends 
from pytrends.request import TrendReq
pytrend = TrendReq()


stockprices = []
unique_stockname =[]
trendarray = []
googletrendskeywords = []

def get_google_trends_data():
    for i in unique_stockname:
        googletrendskeywords.append(i)
        keyword_codes=[pytrend.suggestions(keyword=i)[0] for i in googletrendskeywords]
        df_codes =pd.DataFrame(keyword_codes)
        exact_keywords = df_codes['mid'].to_list()
        country = ["US"]
        category = 107
        search_type=""
        Individual_EXACT_KEYWORD = list(zip(*[iter(exact_keywords)]*1))
        Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
        dicti = {}
        for i in country:
            for keyword in Individual_EXACT_KEYWORD:
            pytrend.build_payload(kw_list=keyword, timeframe = 'today 1-y', geo = country, cat=category, gprop=search_type) 
            dicti[i] = pytrend.interest_over_time()
        df_trends = pd.concat(dicti, axis=1)




def get_stock_prices(stockname):
    stockname = stockname.upper()
    URL = "https://finance.yahoo.com/quote/" + uniquestock + "/history"
    #currently defaults to exactly one year behind the current date, need to implement feature to dynamically find beginning of each company's price history so that max amount of data can be used for predictions
    page = requests.get(URL)
    soup = BeautifulSoup(page.text, "html.parser")
    goodsoup = soup.prettify()
    endofdaystockprices = goodsoup.find_all('td', attrs = {'class':'ts5'})
    #might need ts0 and change row[0] to row[5]
    for i in endofdaystockprices:
        row = i.find_all('td')
        stockprices.append(row[0].text.strip())

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# part of the program where it takes in the file input which contains all the specified stocks, and iterates through the process of getting stock prices for each one + calculating regression & predicting next day stock values
stocknames = open("differentstocks.txt", "r")
for line in stocknames:
    line = line.strip('\n')
    unique_stockname.append(line)
trenddata = open("multiTimelineAPPL.txt", "r");
count = file_len(trenddata)
for i in range(count):
    line = trenddata.readline()
    trendarray.append(line) #needs to be doublechecked to ensure that this is actually selecting the proper values from the 
regressiondata = pd.DataFrame(columns = 'stockprices', 'trendarray')


while stocknames:
    uniquestock = stocknames.readline()
    get_stock_prices(uniquestock)
    
    




