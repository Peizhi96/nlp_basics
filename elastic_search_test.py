from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
def test_elasticsearch_connection():
    # Check if the connection is successful
    if es.ping():
        print("Elasticsearch connection is successful.")
    else:
        print("Elasticsearch connection failed.")
        
test_elasticsearch_connection()