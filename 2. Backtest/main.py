import RF_Algo_Backtest_07172024
import VWAP_Algo_Backtest_07152024
import login
import pyfiglet

#call the login module to log into Robinhood
login.main()

#Define the tickers of interest
stock_ticker = ["TSLA","AAPL"]

#Define the time periods of interest
# options - 5minute”, “10minute”, “hour”, “day”
aggregation_window = ["day"]

#Define the time period shorter than the time period of interest
# options - 5minute”, “10minute”, “hour”, “day”
shorter_aggregation_window = ["hour"]
#full_data_span
full_data_span = ["5year"]

def printing_method_name(text):
    text = pyfiglet.figlet_format(text)
    return text

    

if __name__ == '__main__':
    
  for i in range(len(stock_ticker)):
      for j in range(len(aggregation_window)):

        print(printing_method_name("Random Forest Backtest for "+stock_ticker[i]+" and interval "+aggregation_window[j]))    
        method ="Random Forest"
        RF_Algo_Backtest_07172024.main(stock_ticker[i],aggregation_window[j],shorter_aggregation_window[j],method,full_data_span[j])
        

        method ="VWAP"
        print(printing_method_name("VWAP  backtest result for "+stock_ticker[i]+" and interval "+aggregation_window[j]))
        VWAP_Algo_Backtest_07152024.main( stock_ticker[i],aggregation_window[j],shorter_aggregation_window[j],method,full_data_span[j])

        
    

