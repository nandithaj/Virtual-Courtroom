from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    """Test MongoDB connection and insert/query operations."""
    print("Testing MongoDB connection...")
    
    # Get MongoDB URI from environment
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("ERROR: MONGO_URI environment variable not set!")
        return False
        
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client["gd"]
        test_collection = db["test_collection"]
        
        # Test server info
        server_info = client.server_info()
        print(f"MongoDB version: {server_info.get('version', 'unknown')}")
        
        # Test write operation
        test_doc = {
            "test_id": "mongodb_test",
            "timestamp": datetime.datetime.utcnow(),
            "message": "MongoDB connection test"
        }
        
        result = test_collection.insert_one(test_doc)
        print(f"Test document inserted with ID: {result.inserted_id}")
        
        # Test read operation
        found_doc = test_collection.find_one({"test_id": "mongodb_test"})
        if found_doc:
            print(f"Successfully retrieved test document: {found_doc['_id']}")
            
            # Test update operation
            update_result = test_collection.update_one(
                {"_id": found_doc["_id"]},
                {"$set": {"updated": True}}
            )
            print(f"Updated document: {update_result.modified_count} document(s) modified")
            
            # Test delete operation
            delete_result = test_collection.delete_one({"_id": found_doc["_id"]})
            print(f"Deleted document: {delete_result.deleted_count} document(s) deleted")
            
            return True
        else:
            print("ERROR: Could not find the inserted test document!")
            return False
            
    except Exception as e:
        print(f"ERROR connecting to MongoDB: {e}")
        return False

if __name__ == "__main__":
    success = test_mongodb_connection()
    if success:
        print("MongoDB connection test SUCCESSFUL! ✅")
    else:
        print("MongoDB connection test FAILED! ❌") 