import os
import faiss
import json
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
import fitz  # Import fitz from PyMuPDF

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(device)  # Move model to GPU if available

# Create a SentenceTransformer model using the loaded AutoModel and AutoTokenizer
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
sentence_model.to(device)  # Move SentenceTransformer model to GPU if available

# Display the number of dimensions the model supports for the vector database
embedding_dim = sentence_model.get_sentence_embedding_dimension()
print(f"The model supports {embedding_dim} dimensions for the vector database.")

# Define the data directory
data_dir = "data"

if os.path.exists(data_dir):
    # Iterate over files in the directory
    for filename in os.listdir(data_dir):
        # Print each filename
        print(filename)
else:
    print(f"Directory '{data_dir}' does not exist.")

def read_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(directory, filename)
            doc = fitz.open(doc_path)
            num_pages = doc.page_count
            text = [doc[i].get_text() for i in range(num_pages)]
            documents.append({"source": filename, "text": text})
            doc.close()
    return documents

# Load PDFs
documents = read_pdfs(data_dir)

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False)

# Split documents into chunks
all_chunks = []
metadata = []

for doc in documents:
    doc_name = doc["source"]
    for page_num, page_content in enumerate(doc["text"]):
        chunks = text_splitter.split_text(page_content)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"pdf_name": doc_name, "pdf_page": page_num, "chunk_index": chunk_index})

# Number of chunks
total_chunks = len(all_chunks)
print(f"Total chunks: {total_chunks}")

# Create embeddings and build the FAISS index
index = faiss.IndexFlatL2(embedding_dim)

for i, chunk in enumerate(all_chunks):
    # Tokenize the chunk
    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

    # Pass the inputs through the model to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pool the embeddings
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU for FAISS

    # Add the vector to the index
    index.add(embeddings)

    # Print progress
    print(f"Processing chunk {i + 1} out of {total_chunks}", end="\r")

# Save the index
faiss.write_index(index, "vector_index.faiss")

# Save the metadata
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

print("\nVector database creation complete.")

