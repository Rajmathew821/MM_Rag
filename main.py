from pipeline import run_rag_pipeline

if __name__ == "__main__":
    PDF_PATH = "attention.pdf"
    QUERY = "explain Scaled Dot-Product Attention"
    result = run_rag_pipeline(PDF_PATH, QUERY)
    print("\n🧠 GPT-4o Answer:\n", result["answer"])
    if result["matched_images"]:
        print("\n🖼 Related Image(s):")
        for url in result["matched_images"]:
            print(url)
    else:
        print("\nNo matching images found.")
