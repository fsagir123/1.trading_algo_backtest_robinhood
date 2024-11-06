import RF_Algo_Backtest_07172024
import VWAP_Algo_Backtest_07152024
import login
import pyfiglet
import pandas as pd
import matplotlib.pyplot as plt


#call the login module to log into Robinhood
login.main()

#Define the tickers of interest
stock_ticker = ["NFE","PLUG"]
#Define the time periods of interest
# options - 5minute”, “10minute”, “hour”, “day”
aggregation_window = ["day"]
#Define the time period shorter than the time period of interest
# options - 5minute”, “10minute”, “hour”, “day”
shorter_aggregation_window = ["hour"]
#full_data_span
full_data_span = ["5year"]

def print_method_name(text):
    text = pyfiglet.figlet_format(text)
    return text

def save_results_to_excel(results, filename):
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Stock Ticker', 'Algorithm',"methodology", 'Initial Balance', 'Final Balance', 'Cumulative Returns (%)',"Total Buy Signals"])    
    # Save to Excel
    results_df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")    

if __name__ == '__main__':
  results = []
  # Clear plot area or close all open plots
  plt.close('all')  # This will close any previously opened figures
    
  for stock in stock_ticker:
        for window in aggregation_window:

            # ML Backtest
            print_method_name(f"ML Backtest for {stock} and interval {window}")
            ml_results = RF_Algo_Backtest_07172024.main(stock, window, shorter_aggregation_window[0], 'ML', full_data_span[0])
            results.extend(ml_results)
            

            # VWAP Backtest
            print_method_name(f"VWAP Backtest for {stock} and interval {window}")
            vwap_results = VWAP_Algo_Backtest_07152024.main(stock, window, shorter_aggregation_window[0], 'Algo', full_data_span[0])
            results.append(vwap_results)

    # Save results to Excel
  save_results_to_excel(results, 'backtest_results.xlsx')

