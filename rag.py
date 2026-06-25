from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Step 1: Loading PDF...")
loader = PyPDFLoader("notes.pdf")
pages = loader.load()
print(f"  Loaded {len(pages)} pages")

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"  Created {len(chunks)} chunks")

print("Step 3: Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Step 4: Building FAISS vector database...")
vectordb = FAISS.from_documents(chunks, embeddings)
vectordb.save_local("faiss_index")
print("  Vector DB saved!")

print("Step 5: Loading QA model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

print("\n--- RAG Pipeline Ready! ---")
print("Type your question (or 'quit' to exit)\n")

while True:
    query = input("Your question: ").strip()
    if query.lower() == "quit":
        break
    docs = vectordb.similarity_search(query, k=3)
    context = " ".join([d.page_content for d in docs])[:1000]
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nAnswer: {answer}")
    print(f"Source page: {docs[0].metadata.get('page', 'N/A')}\n")