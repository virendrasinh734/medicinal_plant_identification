import json
with open("plantjson.json","r") as file:
    data=json.load(file)
    print(data[0])