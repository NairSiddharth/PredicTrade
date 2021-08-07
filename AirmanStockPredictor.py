import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  
import pytrends 
from pytrends.request import TrendReq
from numba import jit


pytrend = TrendReq()
stockprices = {}
unique_stocknames_tickers = []
unique_stocknames_names = []
cleaned_ptfr_alltime ={}
trendarray = []
googletrendskeywords = []
df_trends = pd.DataFrame()
ptfcf_dataframe=  pd.DataFrame
stockprices_dataframe = pd.DataFrame()


def get_google_trends_data():
    for i in unique_stocknames_tickers:
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
    df_trends.columns = unique_stocknames_names
    #np.savetxt('GoogleTrendDataStocks.txt', df_trends, fmt='%d')
    #df_trends.to_csv('GoogleTrendDataStocks.txt',)

# NOTE - instead of google trends data, maybe use companes liquid cash flow as a regression variable because it could indicate the health of the company *possibly could have weak correlation to stock price as liquid cash flow does not beenfit stock holders unless actively used, which it typically is not. 
# FURTHER NOTE - have decided to incorporate an additional variable into the regression rather than just google trend data & price, this third variable will be price to free-cash-flow ratio
# FURTHER FURTHER NOTE - in future, may decide to either change said variable or add an additional variable to/using the price/book ratio of a stock

#scrapes price to free cash flow data from this handy website, makes the data usable using beautiful soup, then reads through it + appends all the necessary values to an array/list/dataframe
def price_to_fcf_ratio(stockname_actual,stockname_ticker):
    URL = "https://www.macrotrends.net/stocks/charts/" + stockname_ticker + "/" + stockname_actual + "/price-fcf"
    page = requests.get(URL)
    soup = BeautifulSoup(page.text, "html.parser")
    goodsoup = soup.prettify()
    price_to_fcf_ratios_alltime = goodsoup.find_all('td', attrs = {'class':'tr5'})
    for i in price_to_fcf_ratios_alltime:
        row = i.find_all('td')
        cleaned_ptfr_alltime.append(row[0].text.strip())
    ptfcf_dataframe = pd.concat(cleaned_ptfr_alltime, axis=1)
    ptfcf_dataframe.columns = ptfcf_dataframe.columns.droplevel(0) #drop outside header
    ptfcf_dataframe.reset_index(level=0,inplace=True)
    ptfcf_dataframe.columns = unique_stocknames_names

def get_stock_prices(stockname_ticker):
    stockname = stockname_ticker.upper()
    URL = "https://finance.yahoo.com/quote/" + stockname + "/history"
    #currently defaults to exactly one year behind the current date, need to implement feature to dynamically find beginning of each company's price history so that max amount of data can be used for predictions
    page = requests.get(URL)
    soup = BeautifulSoup(page.text, "html.parser")
    goodsoup = soup.prettify()
    endofdaystockprices = goodsoup.find_all('td', attrs = {'class':'ts5'})
    #might need ts0 and change row[0] to row[5]
    for i in endofdaystockprices:
        row = i.find_all('td')
        stockprices[i] = row[0]
    stockprices_dataframe = pd.concat(stockprices, axis=1)
    stockprices_dataframe.columns = stockprices_dataframe.columns.droplevel(0)
    stockprices_dataframe.reset_index(level=0,inplace=True)
    stockprices_dataframe.columns = unique_stocknames_names


# might be unnecessary for same reasons as clean_trenddata
@jit(nopython=True)
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# part of the program where it takes in the file input which contains all the specified stocks, and iterates through the process of getting stock prices for each one + calculating regression & predicting next day stock values
@jit(nopython=True)
def clean_stocknames_tickers():
    stocknames_tickers = open("differentstocks-tickers.txt", "r")
    for line in stocknames_tickers:
        line = line.strip('\n')
        unique_stocknames_tickers.append(line)

#simple function that goes through the selected stock names and strips the newline character from them so that it can properly be used for the other functions
@jit(nopython=True)
def clean_stocknames_names():
    stocknames_actual = open("stocknames-actual.txt", "r")
    for line in stocknames_actual:
        line = line.strip('\n')
        unique_stocknames_names.append(line)

#may ultimately be unnecessary given that you can just grab a column from a dataframe given that it has a column title, dataframe needs to be tested further to see if has the necessary data to be considered extraneous
@jit(nopython=True)
def clean_trenddata():
    trenddata = open('GoogleTrendDataStocks.txt', "r");
    count = file_len(trenddata)
    for i in range(count):
        line = trenddata.readline()
        trendarray.append(line)

# bulk of the program lies in this function that utilizes the previously accumulated data, concatenates into a dataframe, and then trains a ml model using the random forest regressor to better predict a stock's future (tomorrow's) price. Using random forest allows for error to be minimized compared to other commonly used algorithms for this specific application (i.e. stocks).
def regression(i):
    data = np.array([stockprices_dataframe[unique_stocknames_names[i]], df_trends[unique_stocknames_names[i]], ptfcf_dataframe[unique_stocknames_names[i]]])
    regressiondata = pd.DataFrame(data, columns =['stockprices', df_trends[unique_stocknames_names[i], 'price to free cash flow ratio']])
    x,y,z = regressiondata(return_x_y_z = True) 
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z,  test_size = .70)
    stockpredictor = RandomForestRegressor(n_estimators = 500) #changed from classifier to predictor, unsure what change will do currently > seemed to be consensus choice for this type of program in reputable programming websites
    # after initial few testruns, try out gradient boosted trees to crosscheck different types of ml algorithms and their respective "Accuracy" scores for this data + prediction idea
    stockpredictor.fit(x_train, np.loglp(y_train), np.loglp(z_train))
    y_pred = regressiondata.predict(x_test, z_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    print(unique_stocknames_names[i] + ":c" + stockpredictor(stockprices) + "\n")

#main function to run all other functions in order
def main():
    clean_stocknames_tickers()
    clean_stocknames_names()
    get_google_trends_data()
    for i in range(unique_stocknames_tickers):
        get_stock_prices(unique_stocknames_tickers[i])
        price_to_fcf_ratio(unique_stocknames_names[i],unique_stocknames_tickers[i])
        regression(i)

if __name__ == "__main__":
    main()

    





