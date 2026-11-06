from pymongo import MongoClient

mongo_client = MongoClient("mongodb://127.0.0.1:27017")
db = mongo_client["database"]
db.create_collection("t6")
db["t6"].insert_one({"a":1,"b":2,"c":3})
t1 = db["t5"].find()
print(t1)
