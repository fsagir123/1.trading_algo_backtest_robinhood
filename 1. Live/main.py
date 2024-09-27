import RF_Algo_Live
import login
import pyfiglet
import time
import datetime
import pytz



#call the login module to log into Robinhood
login.main()

#Define the tickers of interest
stock_ticker = ["TSLA","AAPL"]

#Define the time periods of interest
# options - 5minute”, “10minute”, “hour”, “day”
time_period = ["day","hour"]

#Define the time period shorter than the time period of interest
# options - 5minute”, “10minute”, “hour”, “day”
shorter_time_period = ["5minute","5minute"]
#span
span = ["year","3month"]

#method
method ="Random Forest"

# Define the Eastern Time Zone
eastern = pytz.timezone('US/Eastern')

def printing_method_name(text):
    text = pyfiglet.figlet_format(text)
    return text

# Function to check if the current time is a valid execution time
def is_valid_execution_time(current_time):
    # Check if the current day is a weekday (Monday to Friday)
    if current_time.weekday() >= 0 and current_time.weekday() <= 4:
        # Check if the current time is between 9:30 AM and 4:00 PM Eastern Time
        if current_time.time() >= datetime.time(9, 30) and current_time.time() <= datetime.time(16, 0):
            return True
    return False

# Function to check if the current time is within 5 minutes of the valid excecution time
def time_in_5_minute_range_of_valid_excecution_times(current_time,valid_execution_times,valid_execution_times_max):
    for acceptable_time, acceptable_time_max in zip(valid_execution_times,valid_execution_times_max):        
        if current_time.time()>=acceptable_time and current_time.time() <= acceptable_time_max:            
            return True
    return False
   
# Function to run code at specified times
def run_code():
    current_time = datetime.datetime.now(eastern)

    # 9:30 AM to 3:30 PM
    valid_execution_times = [ datetime.time(hour, 30) for hour in range(9, 16)] 
                            
    # 9:35 AM to 3:35 PM
    valid_execution_times_max = [ datetime.time(hour, 35) for hour in range(9, 16)]  
                            
    if time_in_5_minute_range_of_valid_excecution_times(current_time,valid_execution_times,valid_execution_times_max):
        
        # Your code to run goes here
        print("Running code at", current_time.strftime("%I:%M %p"))
        for i in range(len(stock_ticker)): 
            print(printing_method_name("Random Forest trading decision for "+stock_ticker[i]+" and interval "+time_period[1]))
            RF_Algo_Live.main(stock_ticker[i],time_period[1],method,span[1],shorter_time_period[1])



# Main loop to check once every hour
while True:
    print("entered")
    current_time = datetime.datetime.now(eastern)
    # Define the desired wake-up time at 9:30 AM

    if is_valid_execution_time(current_time):
        run_code()

        # Additional code to run only at 3:30 PM
        if current_time.time() >= datetime.time(15,30)  and current_time.time() <= datetime.time(15,35) :
            # Your code to run at 3:30 PM goes here
            print("Running additional code at 3:30 PM")
            for i in range(len(stock_ticker)): 
                    print(printing_method_name("Random Forest trading decision for "+stock_ticker[i]+" and interval "+time_period[0]))
                    RF_Algo_Live.main(stock_ticker[i],time_period[0],method,span[0],shorter_time_period[0])
    
    if current_time.time() < datetime.time(9, 30):
        wake_up_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    # If the current time is after 4 PM, calculate the time until 9:30 AM the next day
    if current_time.hour >= 16:
        # Calculate the time until 9:30 AM the next day
        next_day = current_time + datetime.timedelta(days=1)
        wake_up_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if current_time.time() >= datetime.time(9, 30) and current_time.time() < datetime.time(16, 0):
        if current_time.minute < 30:
           wake_up_time = current_time.replace(minute=30, second=0, microsecond=0)
        if current_time.minute >= 30:
           wake_up_time = current_time.replace(minute=30, second=0, microsecond=0) + datetime.timedelta(hours=1)

    
    # Calculate the time difference between the current time and the wake-up time
    time_difference = (wake_up_time - current_time).total_seconds()+1
    # Put the code to sleep until the specified wake-up time
    print(f"Sleeping until {wake_up_time}")
    time.sleep(time_difference)