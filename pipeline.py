from document_loader import DocumentLoader
from chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from chroma_manager import ChromaManager
from minio_client import MinioClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_API_KEY, CHROMA_DIR

def run_rag_pipeline(pdf_path, user_query):
    # Load and parse document
    loader = DocumentLoader()
    elements = loader.load_pdf(pdf_path)
    text_elements, table_elements, image_elements = loader.separate_elements(elements)

    # Upload images and tables to Minio
    minio = MinioClient()
    for el in table_elements + image_elements:
        if "image_path" in el:
            el["minio_path"] = minio.upload(el["image_path"])

    # Summarize images/tables for embedding
    embedder = EmbeddingGenerator(OPENAI_API_KEY)
    image_prompt = ("You are an assistant tasked with summarizing images for retrieval. "
                    "These summaries will be embedded and used to retrieve the raw image. "
                    "Give a concise summary of the image that is well optimized for retrieval.")
    image_summaries = embedder.summarize_images_and_tables(image_elements + table_elements, image_prompt)

    # Prepare text documents for chunking
    chunker = TextChunker(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50))
    text_summaries = chunker.create_text_documents(text_elements)
    all_summaries = image_summaries + text_summaries

    # Chunk documents and create vectorstore
    text_chunks = chunker.chunk_texts(all_summaries)
    chroma = ChromaManager(OPENAI_API_KEY, CHROMA_DIR)
    db = chroma.build_db(text_chunks)
    retriever = chroma.get_retriever(db)

    # Retrieve context
    context_docs = retriever.get_relevant_documents(user_query, k=10)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Generate answer with prompt
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024, openai_api_key=OPENAI_API_KEY)
    final_prompt = PromptTemplate.from_template("""
        You are the AI assistant to summarize the answer relevant to the context based on user query.
        from the local document: {context}
        question: {query}
    """)
    chain = final_prompt | chat | StrOutputParser()
    response_text = chain.invoke({"context": context_text, "query": user_query})

    # Find and return matched images' URLs
    matched_images = []
    for doc in context_docs:
        doc_id = doc.metadata.get("id", "")
        image_file = f"{doc_id}.png"
        if minio.exists(image_file):
            url = minio.get_presigned_url(image_file)
            matched_images.append(url)

    return {
        "answer": response_text,
        "matched_images": matched_images
    }
