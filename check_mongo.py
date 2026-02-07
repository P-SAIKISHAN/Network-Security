
import os
import sys
import pymongo
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

def check_data():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
        
        # Database and Collection names from your push_data.py
        DATABASE = "Sai_kishan"
        COLLECTION = "NetworkData"
        
        db = client[DATABASE]
        collection = db[COLLECTION]
        
        # Count documents
        count = collection.count_documents({})
        print(f"Success! Connected to database: {DATABASE}")
        print(f"Total documents in '{COLLECTION}' collection: {count}")
        
        # Print a sample document
        if count > 0:
            print("\nSample document:")
            print(collection.find_one())
            
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

if __name__ == "__main__":
    check_data()
