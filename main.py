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

Conversation History:
{history}

Context:
{context}

Medical Conversation History:
{medi}

General Conversation History:
{geno}

Expected Response Type:
{expected}

---
IMPORTANT POINTS:
- If expected response type is related to general conversation, then you just refer to General Conversation History to answer the questions
- If expected reponse type is medically relevant, then you refer to medical conversation history to reform the question asked to add missing details in the question, then pass it on to find contexts, and create an accurate response
- Your answers are supposed to be correct, precise, and should not contain any unnecessary information
- If you do not know about a topic, or if the response is not close to the question asked, then respond by saying that you did not understand the question or that information was missing.
- Answer according to the question asked, but check rephrased question for better answer formulation.

Question asked:
- {question}
Rephrased question
- {q_rephrase}
"""

# In-memory conversation history storage
conversation_history = []
medical_conversation_history = []
general_conversation_history = []
current_reponse = ""

def classify_question(query_text, model):
    # Use your ChatGroq model to classify whether the question is medical or general
    response = model.invoke(f"Just repond with 'medical' if input is medically relevant and respond with 'general' if input is a general conversation question:{query_text}")
    # Assuming the model provides a classification result
    classification_result = response  # You need to define how your model outputs the classification
    global current_reponse
    
    if classification_result == "medical":
        current_reponse="medical"
        return "medical"
    else:
        current_reponse="general"
        return "general"

def main():
    # Initialize your ChatGroq model
    model = ChatGroq(api_key=GROQ_API_KEY, model=MODEL)
    
    while True:
        query_text = input("You: ")
        if query_text == "/exit":
            print("Goodbye!")
            break
        category = classify_question(query_text, model)
        query_rag(query_text, category, model)
        if category == "medical":
            medical_conversation_history.append(f"Client: {query_text}")
        else:
            general_conversation_history.append(f"Client: {query_text}")

def query_rag(query_text: str, category: str, model):
    # Prepare the DB
    embedding_function = CohereEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Select the appropriate conversation history based on category
    if category == "medical":
        current_history = medical_conversation_history
    else:
        current_history = general_conversation_history
    
    # Rephrase the query if it is medically relevant
    if category == "medical":
        # Example of rephrasing: prepend information from medical conversation history
        rephrased_query = " ".join([f"previous medical conversation text here", query_text])
    else:
        rephrased_query = query_text
    
    # Search the DB with the rephrased query
    results = db.similarity_search_with_score(rephrased_query, k=3)

    # Construct the context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Construct the conversation history text
    history_text = "\n".join(conversation_history + current_history)

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the input questions for the prompt
    formatted_prompt = prompt_template.format(
        context=context_text,
        history=conversation_history,
        medi=medical_conversation_history,
        geno=general_conversation_history,
        expected=current_reponse,
        question=query_text,
        q_rephrase=rephrased_query,
    )

    # Get the response from the model
    response = model.invoke(formatted_prompt)
    
    # Extract the content and sources
    response_text = response.content
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Update the main conversation history
    conversation_history.append(f"Client: {query_text}")
    conversation_history.append(f"Response: {response_text}")
    
    # Update the specific conversation history
    current_history.append(f"Client: {query_text}")
    current_history.append(f"Response: {response_text}")
    
    # Format sources
    formatted_sources = "\n".join([f"- {source}" for source in sources])

    # Format the final response
    formatted_response = f"{response_text}"
    
    print(formatted_response)
    print()
    return formatted_response



if __name__ == "__main__":
    main()
