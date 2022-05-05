from tda import auth, client
from webdriver_manager.chrome import ChromeDriverManager
from auth_params import ACCT_NUMBER, API_KEY, CALLBACK_URL

def authenticate():
    token_path = 'token.pickle'
    