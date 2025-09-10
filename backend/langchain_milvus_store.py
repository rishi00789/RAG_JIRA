#!/usr/bin/env python3
"""
LangChain Milvus Vector Store Integration
Custom vector store implementation that integrates Milvus with LangChain's vector store interface
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MilvusVectorStore(VectorStore):
    """Custom Milvus vector store implementation for LangChain"""
    
    def __init__(
        self,
        collection_name: str = "jira_data",
        embedding_function: Optional[Embeddings] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Milvus vector store
        
        Args:
            collection_name: Name of the Milvus collection
            embedding_function: LangChain embedding function
            connection_args: Milvus connection arguments
            **kwargs: Additional arguments
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function or self._get_default_embedding()
        self.connection_args = connection_args or {
            "host": "localhost",
            "port": "19530"
        }
        
        # Connect to Milvus
        self._connect()
        
        # Initialize or get collection
        self.collection = self._get_or_create_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect("default", **self.connection_args)
            logger.info("✅ Connected to Milvus")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _get_default_embedding(self):
        """Get default embedding function using SentenceTransformers"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            logger.warning("⚠️ HuggingFaceEmbeddings not available, using fallback")
            return None
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                logger.info(f"✅ Collection '{self.collection_name}' already exists")
                collection = Collection(self.collection_name)
                collection.load()
                return collection
            
            # Create new collection
            logger.info(f"🔄 Creating new collection '{self.collection_name}'")
            return self._create_collection()
            
        except Exception as e:
            logger.error(f"❌ Error getting/creating collection: {e}")
            raise
    
    def _create_collection(self) -> Collection:
        """Create a new Milvus collection with JIRA-optimized schema"""
        # Define collection schema for JIRA data
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="issue_key", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="issue_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="project_key", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="priority", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="assignee", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="reporter", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="created_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="updated_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(
            fields=fields, 
            description="JIRA issues, comments, and metadata with embeddings"
        )
        
        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        # Load collection
        collection.load()
        
        logger.info(f"✅ Created and loaded collection '{self.collection_name}'")
        return collection
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store"""
        if not documents:
            return []
        
        try:
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            if self.embedding_function:
                embeddings = self.embedding_function.embed_documents(texts)
            else:
                embeddings = self._generate_fallback_embeddings(texts)
            
            # Prepare data for insertion
            data = [
                texts,
                [meta.get('issue_key', '') for meta in metadatas],
                [meta.get('issue_type', '') for meta in metadatas],
                [meta.get('content_type', '') for meta in metadatas],
                [meta.get('project_key', '') for meta in metadatas],
                [meta.get('status', '') for meta in metadatas],
                [meta.get('priority', '') for meta in metadatas],
                [meta.get('assignee', '') for meta in metadatas],
                [meta.get('reporter', '') for meta in metadatas],
                [meta.get('created_date', '') for meta in metadatas],
                [meta.get('updated_date', '') for meta in metadatas],
                embeddings
            ]
            
            # Insert data
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"✅ Added {len(documents)} documents to collection")
            return [str(id) for id in insert_result.primary_keys]
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Perform similarity search"""
        try:
            # Generate query embedding
            if self.embedding_function:
                query_embedding = self.embedding_function.embed_query(query)
            else:
                query_embedding = self._generate_fallback_embeddings([query])[0]
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=[
                    "text", "issue_key", "issue_type", "content_type", 
                    "project_key", "status", "priority", "assignee", "reporter",
                    "created_date", "updated_date"
                ]
            )
            
            # Convert results to documents
            documents = []
            for result in results[0]:
                entity = result.entity
                metadata = {
                    'issue_key': entity.get('issue_key', ''),
                    'issue_type': entity.get('issue_type', ''),
                    'content_type': entity.get('content_type', ''),
                    'project_key': entity.get('project_key', ''),
                    'status': entity.get('status', ''),
                    'priority': entity.get('priority', ''),
                    'assignee': entity.get('assignee', ''),
                    'reporter': entity.get('reporter', ''),
                    'created_date': entity.get('created_date', ''),
                    'updated_date': entity.get('updated_date', ''),
                    'distance': result.distance
                }
                
                documents.append(Document(
                    page_content=entity.get('text', ''),
                    metadata=metadata
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores"""
        try:
            # Generate query embedding
            if self.embedding_function:
                query_embedding = self.embedding_function.embed_query(query)
            else:
                query_embedding = self._generate_fallback_embeddings([query])[0]
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=[
                    "text", "issue_key", "issue_type", "content_type", 
                    "project_key", "status", "priority", "assignee", "reporter",
                    "created_date", "updated_date"
                ]
            )
            
            # Convert results to documents with scores
            documents_with_scores = []
            for result in results[0]:
                entity = result.entity
                metadata = {
                    'issue_key': entity.get('issue_key', ''),
                    'issue_type': entity.get('issue_type', ''),
                    'content_type': entity.get('content_type', ''),
                    'project_key': entity.get('project_key', ''),
                    'status': entity.get('status', ''),
                    'priority': entity.get('priority', ''),
                    'assignee': entity.get('assignee', ''),
                    'reporter': entity.get('reporter', ''),
                    'created_date': entity.get('created_date', ''),
                    'updated_date': entity.get('updated_date', ''),
                    'distance': result.distance
                }
                
                document = Document(
                    page_content=entity.get('text', ''),
                    metadata=metadata
                )
                
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + result.distance)
                documents_with_scores.append((document, similarity_score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search with score: {e}")
            return []
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """Delete documents by IDs"""
        try:
            # Convert string IDs to integers
            int_ids = [int(id) for id in ids]
            
            # Delete from collection
            self.collection.delete(f"id in {int_ids}")
            self.collection.flush()
            
            logger.info(f"✅ Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.collection.delete("id >= 0")
            self.collection.flush()
            logger.info("✅ Cleared collection")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            return {
                'collection_name': self.collection_name,
                'num_entities': self.collection.num_entities,
                'is_loaded': self.collection.has_index(),
                'index_info': self.collection.index().params if self.collection.has_index() else None
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {}
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate fallback embeddings using hash-based method"""
        try:
            import hashlib
            embeddings = []
            for text in texts:
                # Create a more sophisticated hash-based embedding
                text_hash = hash(text) % 1000000
                md5_hash = hashlib.md5(text.encode()).hexdigest()
                embedding = []
                for i in range(384):
                    char_val = ord(md5_hash[i % len(md5_hash)]) if i < len(md5_hash) else 0
                    embedding.append(float((text_hash + char_val + i) % 100) / 100.0)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Fallback embedding generation failed: {e}")
            # Return random embeddings as last resort
            import random
            return [[random.random() for _ in range(384)] for _ in texts]
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "jira_data",
        **kwargs
    ) -> "MilvusVectorStore":
        """Create vector store from texts"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        return cls.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name=collection_name,
            **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        collection_name: str = "jira_data",
        **kwargs
    ) -> "MilvusVectorStore":
        """Create vector store from documents"""
        vector_store = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            **kwargs
        )
        vector_store.add_documents(documents)
        return vector_store

def create_milvus_vector_store(collection_name: str = "jira_data", **kwargs) -> MilvusVectorStore:
    """
    Factory function to create a MilvusVectorStore instance
    
    Args:
        collection_name: Name of the Milvus collection
        **kwargs: Additional arguments to pass to MilvusVectorStore
        
    Returns:
        MilvusVectorStore instance
    """
    return MilvusVectorStore(
        collection_name=collection_name,
        **kwargs
    )

# Example usage
if __name__ == "__main__":
    # Test the vector store
    try:
        # Create sample documents
        sample_docs = [
            Document(
                page_content="This is a test JIRA issue about implementing user authentication.",
                metadata={
                    'issue_key': 'TEST-1',
                    'issue_type': 'Story',
                    'content_type': 'summary',
                    'project_key': 'TEST',
                    'status': 'To Do',
                    'priority': 'High',
                    'assignee': 'John Doe',
                    'reporter': 'Jane Smith'
                }
            ),
            Document(
                page_content="Need to add password validation and secure login functionality.",
                metadata={
                    'issue_key': 'TEST-1',
                    'issue_type': 'Story',
                    'content_type': 'description',
                    'project_key': 'TEST',
                    'status': 'To Do',
                    'priority': 'High',
                    'assignee': 'John Doe',
                    'reporter': 'Jane Smith'
                }
            )
        ]
        
        # Create vector store
        vector_store = MilvusVectorStore.from_documents(
            documents=sample_docs,
            collection_name="test_jira_data"
        )
        
        # Test similarity search
        results = vector_store.similarity_search("authentication login", k=2)
        
        print(f"\n🔍 Search results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print(f"\n📊 Collection stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error testing vector store: {e}")
