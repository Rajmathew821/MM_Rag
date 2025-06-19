import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from image_utils import compress_and_encode_image

class EmbeddingGenerator:
    def __init__(self, openai_api_key):
        self.chat = ChatOpenAI(model="gpt-4o", max_tokens=1024, openai_api_key=openai_api_key)

    def summarize_images_and_tables(self, elements, prompt):
        summaries = []
        for el in elements:
            if "image_path" in el:
                img_base64 = compress_and_encode_image(el["image_path"])
                summary = self.chat.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ])
                ]).content
                doc = Document(
                    page_content=summary,
                    metadata={
                        "id": el['id'],
                        "image_path": el.get("image_path"),
                        "image_base64": el['metadata'].get('image_base64', ''),
                        "page_number": el['metadata'].get('page_number'),
                        "last_modified": el['metadata'].get('last_modified'),
                        "filename": el['metadata'].get('filename'),
                        "image_mime_type": el['metadata'].get('image_mime_type', '')
                    }
                )
                summaries.append(doc)
        return summaries
