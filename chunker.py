from pathlib import Path
from langchain.schema import Document

class TextChunker:
    def __init__(self, text_splitter):
        self.text_splitter = text_splitter

    def chunk_texts(self, summaries):
        return self.text_splitter.split_documents(summaries)

    def create_text_documents(self, text_elements):
        Path("output/texts").mkdir(parents=True, exist_ok=True)
        docs = []
        for el in text_elements:
            text_file = Path("output/texts") / f"{el['id']}.txt"
            text_file.write_text(el['content'], encoding='utf-8')
            doc = Document(
                page_content=el['content'],
                metadata={
                    "id": el['id'],
                    "page_number": el['metadata'].get('page_number'),
                    "last_modified": el['metadata'].get('last_modified'),
                    "filename": el['metadata'].get('filename')
                }
            )
            docs.append(doc)
        return docs
