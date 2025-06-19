import os
import uuid
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from minio import Minio
from minio.error import S3Error
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

load_dotenv()
OPENAI_API_TOKEN = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

MINIO_ENDPOINT = "192.168.190.214:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "stackmax"
BUCKET_NAME = "mmrag"
DOCUMENT_PATH = "attention.pdf"

minio_client = Minio(MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

elements = partition_pdf(filename=DOCUMENT_PATH, strategy='hi_res', extract_images_in_pdf=True,
                         extract_image_block_types=["Image", "Table"], extract_image_block_to_payload=True)

text_elements, table_elements, image_elements = [], [], []
Path("output/images").mkdir(parents=True, exist_ok=True)
Path("output/tables").mkdir(parents=True, exist_ok=True)

for element in elements:
    doc_id = str(uuid.uuid4())
    element_dict = {"id": doc_id, "metadata": element.metadata.to_dict()}

    if element.category in ['NarrativeText', 'Text']:
        element_dict["content"] = str(element)
        text_elements.append(element_dict)

    elif element.category == 'Table':
        if element.metadata.image_base64:
            img_file = Path("output/tables") / f"{doc_id}.png"
            img_file.write_bytes(base64.b64decode(element.metadata.image_base64.replace('\n', '')))
            element_dict["image_path"] = str(img_file)

        table_elements.append(element_dict)

    elif element.category == 'Image' and element.metadata.image_base64:
        img_file = Path("output/images") / f"{doc_id}.png"
        img_file.write_bytes(base64.b64decode(element.metadata.image_base64.replace('\n', '')))
        element_dict["image_path"] = str(img_file)
        image_elements.append(element_dict)

def compress_and_encode_image(image_path, resize_width=512):
    img = Image.open(image_path)
    if img.width > resize_width:
        ratio = resize_width / img.width
        img = img.resize((resize_width, int(img.height * ratio)))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

prompt = """You are an assistant tasked with summarizing images for retrieval. \
These summaries will be embedded and used to retrieve the raw image. \
Give a concise summary of the image that is well optimized for retrieval."""

from langchain.schema import Document

summaries = []
for element in image_elements + table_elements:
    if "image_path" in element:
        img_base64 = compress_and_encode_image(element["image_path"])
        summary = chat.invoke([HumanMessage(content=[{"type": "text", "text": prompt},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}])]).content
        doc = Document(page_content=summary,
        metadata={
                "id": element['id'],
                "image_path": element["image_path"],
                "image_base64": element['metadata']['image_base64'],
                "page_number": element['metadata']['page_number'],
                "last_modified":element['metadata']['last_modified'],
                "filename": element['metadata']['filename'],
                "image_mime_type": element['metadata']['image_mime_type']
        }
        )
        summaries.append(doc)
                
def upload_to_minio(file_path):
    try:
        object_name = Path(file_path).name
        minio_client.fput_object(bucket_name=BUCKET_NAME, object_name=object_name, file_path=file_path)
        return object_name
    except S3Error as e:
        print(f"MinIO upload error: {str(e)}")
        raise

for element in table_elements + image_elements:
    if "image_path" in element:
        element["minio_path"] = upload_to_minio(element["image_path"])
    if element.get("content") and element in table_elements:
        element["content_minio_path"] = upload_to_minio(element["content"])

Path("output/texts").mkdir(parents=True, exist_ok=True)
for element in text_elements:
    text_file = Path("output/texts") / f"{element['id']}.txt"
    text_file.write_text(element['content'], encoding='utf-8')
    # element["minio_path"] = upload_to_minio(str(text_file))
    doc = Document(page_content=element['content'],
        metadata={
                "id": element['id'],
                "page_number": element['metadata']['page_number'],
                "last_modified":element['metadata']['last_modified'],
                "filename": element['metadata']['filename']
        }
        )
    summaries.append(doc)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

text_chunks=text_splitter.split_documents(summaries)

db_path = "mm_rag2"
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_TOKEN)

texts = [doc.page_content for doc in text_chunks]
metadatas = [doc.metadata for doc in text_chunks]

db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=db_path
)

db.persist()

retriever = db.as_retriever()

query = "explain Scaled Dot-Product Attention"
context_docs = retriever.get_relevant_documents(query, k=10)

context_text = "\n\n".join([doc.page_content for doc in context_docs])

final_prompt = PromptTemplate.from_template("""
    you are the AI assistant to summarize the answer relevant to the context based on user query.
    from the local document: {context}
    question: {query}
""")

chain = final_prompt | chat | StrOutputParser()
response_text = chain.invoke({"context": context_text, "query": query})


from datetime import timedelta
from IPython.display import Image as IPImage, display

def get_presigned_url(bucket, object_name, expiry_seconds=3600):
    return minio_client.presigned_get_object(
        bucket_name=bucket,
        object_name=object_name,
        expires=timedelta(seconds=expiry_seconds)
    )

matched_images = []

for doc in context_docs:
    doc_id = doc.metadata.get("id", "")
    image_file = f"{doc_id}.png"
    try:
        url = get_presigned_url(BUCKET_NAME, image_file)
        minio_client.stat_object(BUCKET_NAME, image_file)
        matched_images.append(url)
    except S3Error:
        continue

# 🧠 Display the answer
print("🧠 GPT-4o Answer:\n")
print(response_text)

if matched_images:
    print("\n🖼 Related Image(s):")
    for url in matched_images:
        try:
            display(IPImage(url=url))
        except Exception:
            continue 
else:
    print("\nNo matching images found.")