"""Retriever for a customer support policy lookup tool.

Starts empty and must be initialized to fetch, embed, and store policy documents.
Provides semantic search over the embedded documents and can generate a LangChain tool.
"""

import re

import numpy as np
import requests
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool


class PolicyRetriever:
    """Retriever interface for policy document search."""

    FAQ_URL = (
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
    )

    def __init__(self, embedding_model: Embeddings | str) -> None:
        """Initialize an empty Retriever.

        Args:
            embedding_model: Either an embedding model instance or a model name string.
        """
        if isinstance(embedding_model, str):
            print(f"Initializing embedding model from name: {embedding_model}")
            self._model = init_embeddings(embedding_model)
        elif isinstance(embedding_model, Embeddings):
            self._model = embedding_model
        else:
            msg = "embedding must be an Embeddings instance or a model name string"
            raise TypeError(msg)

        self._docs: list[dict] | None = None
        self._arr: np.ndarray | None = None

    def initialize(self) -> None:
        """Download FAQ content, split into sections, embed, and store in memory."""
        print(f"Fetching FAQ content from {self.FAQ_URL}...")
        response = requests.get(self.FAQ_URL, timeout=10)
        response.raise_for_status()
        faq_text = response.text

        docs = [
            {"page_content": section} for section in re.split(r"(?=\n##)", faq_text)
        ]
        contents = [doc["page_content"] for doc in docs]

        print("Embedding documents...")
        vectors = self._model.embed_documents(contents)

        self._docs = docs
        self._arr = np.array(vectors)
        print(f"Loaded and embedded {len(docs)} documents.")

    def query(self, query: str, k: int = 5) -> list[dict]:
        """Query the retriever for top-k similar documents.

        Args:
            query (str): Search query.
            k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list[dict]: Top-k documents with similarity scores.

        Raises:
            RuntimeError: If retriever has not been initialized.
        """
        if self._docs is None or self._arr is None:
            # In the tutorial, we don't want to allow a user to
            # run an empty retriever by mistake.
            # So while this is not a good practice in production (as an empty index
            # is a valid state), we raise an error here to prevent misuse.
            msg = (
                "Retriever is not initialized. "
                "Please call retriever.initialize() first."
            )
            raise RuntimeError(msg)

        query_vector = np.array(self._model.embed_query(query))
        scores = query_vector @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]

        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]

    def as_tool(self, k: int = 2) -> BaseTool:
        """Return a LangChain tool for looking up policy information.

        Args:
            k (int, optional): Number of top documents to return in the tool.
            Defaults to 2.

        Returns:
            Callable: A LangChain tool function.
        """
        retriever = self  # capture self in closure

        @tool
        def lookup_policy(query: str) -> str:
            """Consult company policies to check whether certain options are permitted.

            Args:
                query (str): The user query about policy information.

            Returns:
                str: Concatenated content of the most relevant policy documents.
            """
            top_docs = retriever.query(query, k=k)
            return "\n\n".join(doc["page_content"] for doc in top_docs)

        return lookup_policy
