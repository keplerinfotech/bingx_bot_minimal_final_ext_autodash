import os
from binance.spot import Spot as Client

key = os.environ['BINANCE_API_KEY']
secret = os.environ['BINANCE_API_SECRET']

c = Client(key, secret, base_url='https://testnet.binance.vision')

print('Ping:', c.ping())
print('Time:', c.time())

try:
    acct = c.account(recvWindow=60000)
    print('Account OK, balances entries:', len(acct.get("balances", [])))
except Exception as e:
    print('Account call failed:', e)
