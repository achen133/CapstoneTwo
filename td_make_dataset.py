from config import ACCT_NUMBER, API_KEY, CALLBACK_URL, JSON_PATH
from td.client import TDClient

td_client = TDClient(client_id = API_KEY, redirect_uri = CALLBACK_URL, account_number = ACCT_NUMBER, credentials_path = JSON_PATH)
td_client.login()