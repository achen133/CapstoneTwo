from tda import auth, client
from webdriver_manager.chrome import ChromeDriverManager
from config import ACCT_NUMBER, API_KEY, CALLBACK_URL

#connecting to tdameritrade with credentials
def authenticate():
    token_path = 'token.pickle'
    try:
        c = auth.client_from_token_file(token_path, API_KEY)
    except FileNotFoundError:
        from selenium import webdriver
        with webdriver.Chrome(ChromeDriverManager().install()) as driver:
            c = auth.client_from_login_flow(
                driver, API_KEY, CALLBACK_URL, token_path)

