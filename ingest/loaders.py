from langchain_community.document_loaders import (
PyPDFLoader,
TextLoader,
WebBaseLoader,
)
from urllib.parse import urlparse
import os

class MultiSourceLoader:

    def __init__(self):
        pass

    def load(self, inputs: list):
        all_docs = []

        for input_path in inputs:
            if self._is_url(input_path):
                docs = self._load_web(input_path)
            elif input_path.endswith(".pdf"):
                docs = self._load_pdf(input_path)
            elif input_path.endswith(".md") or input_path.endswith(".txt"):
                docs = self._load_text(input_path)
            else:
                print(f"Unsupported type: {input_path}")
                continue

            # Attach metadata
            for doc in docs:
                doc.metadata["source_path"] = input_path
                doc.metadata["source_type"] = self._get_source_type(input_path)

            all_docs.extend(docs)

        return all_docs

    def _load_pdf(self, path):
        loader = PyPDFLoader(path)
        return loader.load()

    def _load_text(self, path):
        loader = TextLoader(path)
        return loader.load()

    def _load_web(self, url):
        loader = WebBaseLoader(url)
        return loader.load()

    def _is_url(self, path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _get_source_type(self, path):
        if self._is_url(path):
            return "web"
        elif path.endswith(".pdf"):
            return "pdf"
        elif path.endswith(".md"):
            return "markdown"
        else:
            return "text"

