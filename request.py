import requests

url = 'http://localhost:8000//predict'

r = requests.post(url,json={'x_predict':{},})
print(r.json())
