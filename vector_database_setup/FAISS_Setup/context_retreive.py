import torch
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for extracting text from PDFs
import faiss
import numpy as np
import json
from transformers import AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import time  # Import the time module

# Load environment variables from .env
load_dotenv()

# Accessing the variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the SentenceTransformer model
print("Loading SentenceTransformer model...")
sentence_model = SentenceTransformer('your-model-name-here', trust_remote_code=True)
sentence_model.to(device)  # Move model to GPU if available

# Initialize the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('your-model-name-here')

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index('your-faiss-index-file.faiss')

# Load the metadata
print("Loading metadata...")
with open('your-metadata-file.json', 'r') as f:
    metadata = json.load(f)

def similarity_search(query, top_k=3):
    # Tokenize the query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

    # Pass the inputs through the model to get embeddings
    with torch.no_grad():
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.unsqueeze(0)  # Ensure it's a 2D array with shape (1, embedding_dim)
        query_embedding = query_embedding.to("cpu").numpy()  # Move embedding back to CPU for FAISS

    # Search the index for the top_k most similar vectors
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the corresponding chunks and their metadata
    results = []
    for i, idx in enumerate(indices[0]):
        # Fetch the chunk metadata
        chunk_metadata = metadata[idx]
        pdf_name = chunk_metadata["pdf_name"]
        pdf_page = chunk_metadata["pdf_page"]
        chunk_index = chunk_metadata["chunk_index"]
        result = {
            "chunk": {
                "pdf_name": pdf_name,
                "pdf_page": pdf_page,
                "chunk_index": chunk_index
            },
            "distance": distances[0][i]
        }
        results.append(result)

    return results

def extract_text_from_page(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    text = doc[page_num].get_text()
    doc.close()
    return text

def generate_response(context, query, last_user_query, last_bot_answer):
    # Initialize the ChatGoogleGenerativeAI model
    google_model = ChatGoogleGenerativeAI(
        model="your-model-name-here",
        google_api_key=GEMINI_API_KEY,
        temperature=0.6
    )

    # Combine context and query into the message history
    domain = [
        "Your domain here",
        # Add more domains as needed
    ]
    ## Feel free to edit the prompt, apply your own Chain-Of-Thought Reasoning Here
    prompt_template = f"""
    1. You are a specialist in the field of your-domain-here.
    2. Your task is to answer people's queries based on the provided context and question.
    3. You answer only if the context fetched contains the answer to the user's question.
    4. Refrain from answering irrelevant or inappropriate questions.
    5. Explain terms in simple and easy-to-understand language.
    6. If the question is outside your domain, do not answer, and explain why.
    7. Provide complete information related to the user's query, only if it is within your domain.
    8. For basic queries, respond briefly:
        8.1. Greetings: 'Hi there!' 😊
        8.2. Gratitude: 'I hope I helped you.' 😊
        8.3. Farewells: 'Alrighty then! Bye Bye' 😊"""

    modified_query = f"""
1. Follow the prompt template: {prompt_template}.
2. User's last query: {last_user_query}.
3. Your answer to the last query: {last_bot_answer}.
4. User's current query: {query}.

Now answer only user's current query, take help from last query to understand the question, if required.
"""

    print(modified_query)

    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": modified_query}
    ]

    # Generate the response
    response = google_model.invoke(messages)

    return response.content  # Correctly extract the response content

def main():
    # Sample input data
    query_text = 'your-query-text-here'
    last_user_query = 'your-last-user-query-here'
    last_bot_answer = 'your-last-bot-answer-here'

    start_time = time.time()  # Record the start time

    try:
        updated_query = f"""Current query: {query_text}; Last query: {last_user_query}; Last answer: {last_bot_answer}"""
        # Perform similarity search to get the top k relevant chunks
        print(f"Performing similarity search for query: {updated_query}")
        results = similarity_search(query_text, top_k=3)

        # Prepare the context from the search results
        context = ""
        visited_pages = set()  # To keep track of visited pages and avoid duplication

        for res in results:
            chunk = res["chunk"]
            pdf_name = chunk["pdf_name"]
            pdf_page = chunk["pdf_page"]
            page_key = (pdf_name, pdf_page)

            if page_key not in visited_pages:
                visited_pages.add(page_key)
                pdf_path = f"your-pdf-folder/{pdf_name}"
                page_text = extract_text_from_page(pdf_path, pdf_page)

                context += f"Document: {pdf_name}, Page: {pdf_page}\n"
                context += f"Text:\n{page_text}\n\n\n"

        print(f"Generated context: {context}")

        # Generate the response
        response = generate_response(context, query_text, last_user_query, last_bot_answer)
        print(f"Generated response: {response}")

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        if response:
            print({'response': response})
        else:
            print({'response': 'Response Not Generated...'})

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()
