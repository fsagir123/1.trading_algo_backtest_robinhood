import RF_Algo_Backtest_07172024
import VWAP_Algo_Backtest_07152024
import login
import pyfiglet
import pandas as pd

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

def save_results_to_excel(results, filename):
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Stock Ticker', 'Algorithm', 'Initial Balance', 'Final Balance', 'Cumulative Returns (%)'])
    
    # Save to Excel
    results_df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")    

if __name__ == '__main__':

  results = []
    
  for i in range(len(stock_ticker)):
      for j in range(len(aggregation_window)):

        print(printing_method_name("ML Backtest for "+stock_ticker[i]+" and interval "+aggregation_window[j]))    
        method ="ML"
        Ensemble_results, LSTM_results = RF_Algo_Backtest_07172024.main(stock_ticker[i],aggregation_window[j],shorter_aggregation_window[j],method,full_data_span[j])
        results.append(Ensemble_results)
        results.append(LSTM_results)

        method ="Algo"
        print(printing_method_name("VWAP  backtest result for "+stock_ticker[i]+" and interval "+aggregation_window[j]))
        VWAP_results = VWAP_Algo_Backtest_07152024.main( stock_ticker[i],aggregation_window[j],shorter_aggregation_window[j],method,full_data_span[j])
        results.append(VWAP_results)
        
        
  # Save all results to an Excel file
  save_results_to_excel(results, 'backtest_results.xlsx')

