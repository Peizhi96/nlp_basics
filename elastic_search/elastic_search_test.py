import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
# Load environment variables from .env file


def test_elasticsearch_connection():
    # Check if the connection is successful
    ELASTIC_HOST = os.getenv('ELASTIC_HOST')
    es = Elasticsearch(ELASTIC_HOST)
    if es.ping():
        print("Elasticsearch connection is successful.")
    else:
        print("Elasticsearch connection failed.")
        
test_elasticsearch_connection()