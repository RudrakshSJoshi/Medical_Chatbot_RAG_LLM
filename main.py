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
You are a medical specialist that provides answers based on the context and finds relevance in questions using conversation history.
Your responses are medically relevant, but you can reply to general questions as well.
You are MediBot, a medical assistance specialist, created on 10th of June, 2024 by a group called Von, an Icelandic term for 'hope', which is what you stand for.

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
- You should always refer to conversation history, no matter what expected reponse type is.
- If expected response type is related to general conversation, then you just refer to General Conversation History to answer the questions.
- If expected reponse type is medically relevant, then you should refer to context and medical conversation history, and answer according to question while also maintaining relevance with the rephrased question.
- Your answers are supposed to be correct, precise, and should not contain any unnecessary information.
- If you do not know about a topic, or if the response is not close to the question asked, then respond by saying that you did not understand the question or that information was missing.
- Answer according to the question asked, but check rephrased question for better answer formulation.
- Your responses should look like you are talking to someone implying it shouldn't contain notes, terms like 'reponse', 'context', 'conversation history', etc.

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
    last_convo =""
    if conversation_history:
        last_convo = " ".join([conversation_history[-1], conversation_history[-2]])
    else:
        last_convo = "the conversation just started"
    # Use your ChatGroq model to classify whether the question is medical or general
    response = model.invoke(f"""
                            Determine whether to respond with 'general' or 'medical' based on the following:
                            - the current conversation text: {query_text}.
                            - the last relevant connversation: {last_convo}.
                            'medical' means the text talks about anything related to medical field, disease, treatment, symptom, health, and such.
                            'general' means the text is not medically relevant, and can be answered independently without context.
                            Your answer should be dependent on both conversations because current conversation may be dependent on previous conversation.
                            Your reply should be a single word.
                            """)
    
    # Extracting the response content
    classification_result = response.content.strip()
    print(f"\n\nTemp: {classification_result}\n\n")
    
    global current_response
    
    if "medical" in classification_result.lower():
        print("\nMedical Term Detected\n")
        current_response = "medical"
        return "medical"
    else:
        print("\nGeneral Term Detected\n")
        current_response = "general"
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
        if medical_conversation_history:
            last_medical_convo = " ".join([medical_conversation_history[-1], medical_conversation_history[-2]])
            rephrased_query = f"Previous medical conversation text and reply here: {last_medical_convo}; Current question: {query_text}"
        else:
            rephrased_query = query_text
        print(f"\n\nRephrasing: {rephrased_query}\n\n")
    else:
        rephrased_query = query_text
    
    # Search the DB with the rephrased query
    results = db.similarity_search_with_score(rephrased_query, k=4)

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
    # formatted_response = f"Response:\n{response_text}\n\nSources:\n{formatted_sources}"
    formatted_response = f"{response_text}"
    
    print(formatted_response)
    print()
    return formatted_response



if __name__ == "__main__":
    main()
