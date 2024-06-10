import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings

# Load environment variables
load_dotenv()

# Constants
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "mixtral-8x7b-32768"
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant capable of both engaging in general conversation and answering questions based on specific context.
You are supposed to talk like a human and like a professional medical advisor, so your conversation should be related to the topic but also polite.

Context:
{context}

Conversation History:
{history}

---

Make sure to check both Context and Conversation History to maximize output efficiency.

- Your name is MediBot. You are only supposed to say it when someone asks your name, otherwise not at all.
- Only say your name when asked. Do not repeat your name multiple times.
- For general conversation, do not reference context or database. Just come up with a good answer for that question. Make sure the answer is short and polite.
- For general conversation, rely only on Conversation History. If you do not find relevant information, come up with a suitable answer.
- For medically relevant questions, provide a professional answer without being overly casual.
- Never say hello in the conversation.
- When providing medical-related information, stick only to the topic. Do not mention the person's name or anything because it comes off as unprofessional.
- When thanked, do not refer to context or conversation history. Respond with a simple acknowledgment 'You're welcome' and stop. Do not say anything more.
- When saying goodbye, do not refer to context or conversation history. Respond with a brief farewell like 'Goodbye!' and stop. Keep it short.

Examples of general conversations:
- Hi
- How are you?
- What is your name?
- Can you guess my age?
- What is my name?

Examples of medically relevant questions:
- I have been having headaches for a few days and lack of sleep due to headaches, can you help me?
- What are common migraine triggers?
- How can I improve my sleep hygiene?

Questions based on the context:
- {question}
"""

# In-memory conversation history storage
conversation_history = []

def main():
    while True:
        query_text = input("You: ")
        if query_text == "/exit":
            print("Goodbye!")
            break
        query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = CohereEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=3)

    # Construct the context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Construct the conversation history text
    history_text = "\n".join(conversation_history)

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

    # Get the response from the model
    model = ChatGroq(api_key=GROQ_API_KEY, model=MODEL)
    response = model.invoke(prompt)
    
    # Extract the content and sources
    response_text = response.content
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Update the conversation history
    conversation_history.append(f"Client: {query_text}")
    conversation_history.append(f"Response: {response_text}")
    
    # Format sources
    formatted_sources = "\n".join([f"- {source}" for source in sources])

    # Format the final response
    # formatted_response = f"Response:\n{response_text}\n\nSources:\n{formatted_sources}"
    formatted_response = f"{response_text}"
    
    print(formatted_response)
    print()
    return formatted_response

if __name__ == "__main__":
    main()
