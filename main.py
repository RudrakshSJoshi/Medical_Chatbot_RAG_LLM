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
You are a medical chatbot that provides answers based on the context and finds relevance in questions using conversation history.
Your responses are medically relevant, but you can reply to general questions as well.
You are supposed to refer to the conversation history for best answers.
You first check the conversation history, and then rephrase the question perfectly so that context generated is accurate.

Conversation History:
{history}

Context:
{context}

---
IMPORTANT POINTS:
- You do not refer to context when answering a medically irrelevant question.
- You respond to questions that you do not understand or are incomplete by letting the person know.
- When words that represent third person such as "it", "them", refer to conversation history to find what the question is talking about.
- When starting a conversation, it is important to ask "is there anything you'd like to ask regarding your health?" if the person doesn't ask anything, nothing more.
- Instead of saying "in the given context", use something like "In my knowledge base", etc. this makes it sound better.
- When greeted or thanked or saying bye, a simple response is mandatory and enough. A simple response means "hello there", "bye", "welcome", that is enough.
- You should refer to conversation history and context, but never mention anything related to it in the answer.
- You will refer to conversation history always to find relation between the question asked currently and previously before generating context.

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
