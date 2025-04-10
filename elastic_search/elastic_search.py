import numpy as np
import json
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import logging 
from gensim.models import FastText 
from tqdm import tqdm

class ElasticSearchVectorEngine:
    def __init__(self, host, index_name):
        self.host = host
        self.index_name = index_name
        self.model = None
        self.vector_dim = 0
        self.es_client = None
        self.logger = self.__setup_logger()
        
        if model_path:
            self.load_model(model_path)
            
    def __setup_logger(self):
        logger = logging.getLogger('ElasticSearchVectorEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self, model_path):
        try:
            self.model = FastText.load(model_path)
            self.vector_dim = self.model.vector_size
            self.logger.info(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
        
    def connect_to_es(self):
        ELASTIC_HOST = os.getenv('ELASTIC_HOST')
        es = Elasticsearch(ELASTIC_HOST)
        if es.ping():
            print("Elasticsearch connection is successful.")
        else:
            print("Elasticsearch connection failed.")
    
    def create_idx(self, drop_existing=False):
        try:
            index_exits = self.es_client.indices.exists(index=self.index_name)
            if drop_existing and index_exits:
                self.es_client.indices.delete(index=self.index_name)
                self.logger.info(f"Deleted existing index: {self.index_name}")
        except Exception as e:
            self.logger.error(f"Error checking index existence: {e}")
            return False

        vector_settings = {
            # dense_vector is used for dense vector fields, dense vector is different from sparse vector
            # if we use Milvus, we need to use sparse vector
            # the differents between dense vector and sparse vector 
            # is that dense vector is used for dense vector fields, sparse vector is used for sparse vector fields
            # milvus uses sparse vector because it is more efficient for sparse vector fields, 
            "type": "dense_vector",
            "dims": self.vector_dim,
        }
        
        index_settings = {
            "settings": {
                # number_of_shards is the number of shards, it represents the number of partitions
                # it is used for load balancing, if the number of shards increases, 
                # it means that the data is more balanced
                "number_of_shards": 1,
                # number_of_replicas is the number of replicas, it represents the number of copies
                # it is used for high availability, if the number of replicas increases, 
                # it means that the data is more secure
                "number_of_replicas": 0,
                # the number of fields in the index
                # it is used for load balancing, if the number of fields increases
                "index.mapping.total_fields.limit": 10000,
            },
            "mappings": {
                "properties": {
                    "id": {
                        "type": "keyword",
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                            }
                        }
                    },
                    "company": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                            }
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "requirements": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "category": {"type": "keyword"},
                    "job_type": {"type": "keyword"},
                    "salary_range": {"type": "keyword"},
                    "experience_level": {"type": "keyword"},
                    "posted_date": {"type": "date"},
                    "is_remote": {"type": "boolean"},
                    
                    "job_vector": vector_settings,
                    "skills": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "level": {"type": "keyword"}
                        }
                    },
                    "benefits": {
                        "type": "nested",
                        "properties": {
                            "type": {"type": "keyword"},
                            "description": {"type": "text"}
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            }
        }
        
        try:
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            self.logger.info(f"Index created: {self.index_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            return False
    
    def generate_embedding(self, text):
        if self.model is None:
            self.logger.error("Model is not loaded.")
            return None
        
        # Check if the text is a string
        words = text.split() if isinstance(text, str) else []
        if not words:
            self.logger.error("Input text is empty or not a string.")
            return np.zeros(self.vector_dim)
        
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not word_vectors:
            self.logger.warning("No words found in the model vocabulary.")
            return np.zeros(self.vector_dim)
        
        # Calculate the mean of the word vectors and normalize
        doc_vector = np.mean(word_vectors, axis=0)
        norm = np.linalg.norm(doc_vector)
        
        if norm > 0:
            doc_vector = doc_vector / norm
            
        return doc_vector.astype(np.float32)
    
    def load_json_data(self, json_file):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            
            self.logger.info(f"Loaded {len(data)} records from {json_file}.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            return None
        
    def preprocess_job_data(self, job_data):
        if self.model is None:
            self.logger.error("Model is not loaded.")
            return None
        
        processed_data = []
        self.logger.info("Preprocessing job data...")
        
        for job in tqdm(job_data, desc="Preprocessing"):
            text_for_vector = ""
            if "title" in job and job["title"]:
                text_for_vector += job["title"] + " "
            if "description" in job and job["description"]:
                text_for_vector += job["description"] + " "
            if "requirements" in job and job["requirements"]:
                text_for_vector += job["requirements"] + " "
            if "skills" in job and isinstance(job["skills"], list):
                for skill in job["skills"]:
                    if isinstance(skill, dict) and "name" in skill:
                        text_for_vector += skill["name"] + " "
                    elif isinstance(skill, str):
                        text_for_vector += skill + " "
                        
            job_vector = self.generate_embedding(text_for_vector)
            job_with_vector = job.copy()
            job_with_vector["job_vector"] = job_vector.tolist()
            
            processed_data.append(job_with_vector)
        self.logger.info("Preprocessing completed.")
        return processed_data
    
    def index_job(self, job_data, batch_size=100):
        if self.es_client is None:
            self.logger.error("Elasticsearch client is not connected.")
            return False
        if not job_data:
            self.logger.error("No job data to index.")
            return False
        self.logger.info("Indexing job data..., batch size: %d", batch_size)
        
        from elasticsearch.helpers import bulk
        
        # prepare the data for bulk indexing
        def create_actions():
            for job in job_data:
                if id not in job:
                    # generate a unique id for the job
                    job["id"] = str(hash(json.dumps(job, sort_keys=True)))
                
                action = {
                    "_index": self.index_name,
                    "_id": job["id"],
                    "_source": job
                }
                
                yield action
        try: 
            # Perform bulk indexing
            success, failed = bulk(
                self.es_client,
                create_actions(),
                chunk_size=batch_size,
                max_retries=5,
                initial_backoff=2,
                request_timeout=60,
            )
            self.logger.info(f"Indexed {success} documents successfully, {failed} failed.")
            self.es_client.indices.refresh(index=self.index_name)
            return success > 0
        except Exception as e:
            self.logger.error(f"Error indexing job data: {e}")
            return False
    
    def vector_search(self, query_text=None, query_vector=None, filters=None, top_k=10):
        """_summary_

        Args:
            query_text (str): query text
            query_vector (list): query vector
            filters (dict): filter conditions
            top_k (int): the number of returned results
        
        return:
            list: list of search results
        """
        if self.es_client is None:
            self.logger.error("Elasticsearch client is not connected.")
            return None
        
        if query_vector is None and query_text is not None:
            query_vector = self.generate_embedding(query_text)
            if query_vector is None or np.all(query_vector == 0):
                self.logger.error("Failed to generate query vector.")
                return None
            
        if query_vector is None:
            self.logger.error("No query vector or text provided.")
            return None 
        
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
            
        script_query = {
            "script_score":{
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['job_vector']) + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
        
        # add filter conditions
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
            
            query = {
                "bool": {
                    "must": script_query,
                    "filter": filter_clauses
                }
            }
        else:
            query = script_query
            
        search_query = {
            "size": top_k,
            "query": query,
            "_source": {
                "excludes": ["job_vector"]
            }
        }
        
        try:
            results = self.es_client.search(index=self.index_name, body=search_query)
            return self.__format__search_results(results)
        except Exception as e:
            self.logger.error(f"Error performing vector search: {e}")
            return None
        
    def __format__search_results(self, es_results):
        """_summary_

        Args:
            results (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not es_results or 'hits' not in es_results or 'hits' not in es_results['hits']:
            return []
        
        formatted_results = []
        hits = es_results['hits']['hits']
        
        for hit in hits:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def hybrid_search(self, query_text, vector_weight=0.7, keyword_weight=0.3, filters=None, top_k=10):
        """
        执行混合搜索（向量 + 关键词）
        
        参数:
            query_text (str): 查询文本
            vector_weight (float): 向量搜索权重
            keyword_weight (float): 关键词搜索权重
            filters (dict): 过滤条件
            top_k (int): 返回结果数量
        
        返回:
            list: 搜索结果
        """
        if self.es_client is None:
            self.logger.error("请先连接到Elasticsearch")
            return None
        
        # 生成查询向量
        query_vector = self.generate_embedding(query_text)
        
        # 构建混合查询
        query_obj = {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            # 关键词匹配部分
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["title^3", "description", "requirements", "skills.name^2"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "functions": [
                    # 向量相似度部分
                    {
                        "script_score": {
                            "script": {
                                "source": f"{vector_weight} * (cosineSimilarity(params.query_vector, doc['job_vector']) + 1.0)",
                                "params": {"query_vector": query_vector.tolist()}
                            }
                        }
                    },
                    # 关键词匹配的分数权重
                    {
                        "weight": keyword_weight,
                        "filter": {"match_all": {}}
                    }
                ],
                "score_mode": "sum"
            }
        }
        
        # 添加过滤条件
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
            
            query_obj["function_score"]["query"]["bool"]["filter"] = filter_clauses
        
        # 完整的搜索请求
        search_query = {
            "size": top_k,
            "query": query_obj,
            "_source": {"excludes": ["job_vector"]}
        }
        
        try:
            results = self.es_client.search(
                index=self.index_name,
                body=search_query
            )
            
            return self._format_search_results(results)
        except Exception as e:
            self.logger.error(f"混合搜索失败: {e}")
            return None

# 用法示例
def main():
    # 初始化引擎
    engine = ElasticSearchVectorEngine(
        host='localhost',
        port=9200,
        index_name='jobs_index',
        model_path='path/to/fasttext/model.bin'
    )
    
    # 连接到Elasticsearch
    if not engine.connect_to_elasticsearch():
        return
    
    # 创建索引
    engine.create_index(drop_existing=True)
    
    # 加载JSON数据
    jobs = engine.load_json_data('jobs.json')
    if not jobs:
        return
    
    # 预处理并索引数据
    processed_jobs = engine.preprocess_job_data(jobs)
    engine.index_jobs(processed_jobs)
    
    # 向量搜索示例
    results = engine.vector_search(
        query_text="machine learning engineer with python experience",
        filters={"job_type": "full-time", "is_remote": True},
        top_k=5
    )
    
    # 混合搜索示例
    hybrid_results = engine.hybrid_search(
        query_text="machine learning engineer with python experience",
        vector_weight=0.7,
        keyword_weight=0.3,
        filters={"location": "San Francisco"},
        top_k=10
    )
    
    # 打印结果
    print("向量搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['source']['title']} (Score: {result['score']:.4f})")
    
    print("\n混合搜索结果:")
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result['source']['title']} (Score: {result['score']:.4f})")

if __name__ == "__main__":
    main()

            
            
        