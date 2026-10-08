from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_DIR = Path("internal_documents")
CHROMA_DIR = "./chroma_db"

embeddings = OllamaEmbeddings(model="llama3.2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

texts, metadatas = [], []
for txt_file in DOCS_DIR.glob("*.txt"):
    content = txt_file.read_text(encoding="utf-8")
    chunks = splitter.split_text(content)
    texts.extend(chunks)
    metadatas.extend([{"source": txt_file.name}] * len(chunks))

if not texts:
    print("No .txt files found in internal_documents/")
else:
    Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=CHROMA_DIR)
    print(f"Ingested {len(texts)} chunks from {DOCS_DIR}")
