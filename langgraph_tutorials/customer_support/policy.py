"""In memory retriever for a customer support policy lookup tool.

This module provides functionality to retrieve relevant policy information based on customer queries
using vector embeddings for semantic search. It fetches policy documents from an external source
and creates a vector store for efficient retrieval.
"""

import re

import numpy as np
import openai
import requests
from langchain_core.tools import tool

# Fetch policy documents from external source
response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

# Split the text into separate documents by section headers
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    """Vector-based retrieval system for finding relevant policy documents.
    
    This class creates and manages embeddings for policy documents and provides
    methods to query the documents based on semantic similarity.
    
    Attributes:
        _arr (numpy.ndarray): Array of document embeddings.
        _docs (list): List of document dictionaries.
        _client (openai.Client): OpenAI client for creating embeddings.
    """

    def __init__(self, docs: list, vectors: list, oai_client) -> None:
        """Initialize the VectorStoreRetriever with documents and their embeddings.
        
        Args:
            docs (list): List of document dictionaries.
            vectors (list): List of embedding vectors corresponding to the documents.
            oai_client: OpenAI client instance for creating embeddings.
        """
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        """Create a VectorStoreRetriever from documents by generating embeddings.
        
        Args:
            docs: List of document dictionaries with a 'page_content' key.
            oai_client: OpenAI client instance for creating embeddings.
            
        Returns:
            VectorStoreRetriever: A new instance with the documents and their embeddings.
        """
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        """Query the vector store for documents similar to the query.
        
        Args:
            query (str): The query string to search for.
            k (int, optional): Number of top results to return. Defaults to 5.
            
        Returns:
            list[dict]: List of document dictionaries with added similarity scores,
                sorted by relevance.
        """
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


# Create a global retriever instance for policy lookups
retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    
    Use this function to search for relevant policy information before making any flight 
    changes or performing other 'write' events.
    
    Args:
        query (str): The user query about policy information.
        
    Returns:
        str: Concatenated content of the most relevant policy documents.
    """
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])
