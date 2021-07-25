import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  
import pytrends 
from pytrends.request import TrendReq
pytrend = TrendReq()


stockprices = []
unique_stocknames =[]
trendarray = []
googletrendskeywords = []

def get_google_trends_data():
    for i in unique_stocknames:
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
    df_trends.columns = df_trends.columns.droplevel(0) #drop outside header
    df_trends = df_trends.drop('isPartial', axis = 1) #drop "isPartial"
    df_trends.reset_index(level=0,inplace=True) #reset_index
    df_trends.columns = unique_stocknames
    #np.savetxt('GoogleTrendDataStocks.txt', df_trends, fmt='%d')
    #df_trends.to_csv('GoogleTrendDataStocks.txt',)

# NOTE - instead of google trends data, maybe use companes liquid cash flow as a regression variable because it could indicate the health of the company *possibly could have weak correlation to stock price as liquid cash flow does not beenfit stock holders unless actively used, which it typically is not. 

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

# might be unnecessary for same reasons as clean_trenddata
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# part of the program where it takes in the file input which contains all the specified stocks, and iterates through the process of getting stock prices for each one + calculating regression & predicting next day stock values
def clean_stocknames():
    stocknames = open("differentstocks.txt", "r")
    for line in stocknames:
        line = line.strip('\n')
        unique_stocknames.append(line)

#may ultimately be unnecessary given that you can just grab a column from a dataframe given that it has a column title, dataframe needs to be tested further to see if has the necessary data to be considered extraneous
def clean_trenddata():
    trenddata = open('GoogleTrendDataStocks.txt', "r");
    count = file_len(trenddata)
    for i in range(count):
        line = trenddata.readline()
        trendarray.append(line) #needs to be doublechecked to ensure that this is actually selecting the proper values from the 


def main():
    clean_stocknames()
    get_google_trends_data()
    for i in range(unique_stocknames): 
        regressiondata = pd.DataFrame(columns =['stockprices', df_trends[unique_stocknames[i]]])
        x,y = regressiondata(return_x_y = True) 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .70)
        stockpredictor = RandomForestClassifier(n_estimators = 100)
        stockpredictor.fit(x_train,y_train)
        y_pred = regressiondata.predict(x_test)
        print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
        print(unique_stocknames[i] + ":c" + stockpredictor(stockprices) + "\n")

    
    
    




