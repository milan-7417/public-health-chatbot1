import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

PDF_FOLDER = "backend/data/pdfs"
VECTOR_FOLDER = "backend/vector_db"

model = SentenceTransformer("BAAI/bge-small-en")

documents = []




for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)

        try:
            reader = PdfReader(pdf_path, strict=False)

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    documents.append(text)

        except Exception as e:
            print(f"Error reading {file}: {e}")

embeddings = model.encode(
    documents,
    batch_size=32,          
    show_progress_bar=True  
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

os.makedirs(VECTOR_FOLDER, exist_ok=True)

faiss.write_index(index, f"{VECTOR_FOLDER}/faiss.index")

np.save(f"{VECTOR_FOLDER}/documents.npy", documents)

print("Vector database created successfully")
