import MetaTrader5 as mt5
from datetime import datetime

#import mt5 account_details.txt
with open('account_details.txt', mode='r') as account_file:
    lines = account_file.read().split('\n')
    account = {}
    for line in lines:
        if ':' in line:
            details = "".join(line.split()).split(':')
            key, value = details[0], details[1]
            account[key] = value
            
mt5_user = account['Login']

#connecting to mt5 servers w/ credentials
def connect(credentials):
    credentials = int(credentials)
    mt5.initialize()
    authorized = mt5.login(credentials, account['Password'], account['Server'])

    #validate authorization
    if authorized:
        print("Connected: Connecting to MT5 Client")
    else:
        print("Failed to connect at account #{}, error code: {}".format(mt5_user, mt5.last_error()))

connect(mt5_user)

#start/stop date parameters 
utc_from = datetime(2021, 1, 1)
utc_to = datetime(2021, 1, 10)

#pulling data
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H4, utc_from, utc_to)
for rate in rates:
    print(rate)
