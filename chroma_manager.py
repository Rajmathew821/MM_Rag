from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class ChromaManager:
    def __init__(self, api_key, persist_dir):
        self.api_key = api_key
        self.persist_dir = persist_dir

    def build_db(self, text_chunks):
        texts = [doc.page_content for doc in text_chunks]
        metadatas = [doc.metadata for doc in text_chunks]
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_dir
        )
        db.persist()
        return db

    def get_retriever(self, db):
        return db.as_retriever()
