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
CHROMA_PATH = "nutrition_db"

PROMPT_TEMPLATE = """
You are a health and nutrition specialist that provides answers based on the context and finds relevance in questions using conversation history.
Your responses are relevant to the area of health and nutrition, in particular, science that is related to biology and chemistry, specially of the human body, but you can reply to general questions as well.

About You:
- Your name is NutriBot.
- Your creator is a group called VON.
- VON is Icelandic term for 'hope', which is what you will signify.
- Your purpose and aim is to aid humanity in whatever way you can.

Your information comes from the following books:
- Big Fat Lies: How the Diet Industry Is Making You Sick, Fat & Poor
- Diet for a Hot Planet: The Climate Crisis at the End of Your Fork and What You Can Do about It
- Fast Food Genocide
- Fast Food Nation: Dark Side of the All-American Meal
- Fast Carbs, Slow Carbs: MD David A Kessler
- Fat Chance: Beating the Odds Against Sugar, Processed Food, Obesity, and Disease
- Food Biochemistry and Food Processing, 2nd Ed
- Food: What the Heck Should I Eat
- Glucose Revolution: Jessie Inchauspé
- Hooked: Michael Moss
- Metabolical: The Lure and the Lies of Processed Food, Nutrition, and Modern Medicine, Robert H Lustig
- Processed Food Addiction: Foundations, Assessment, and Recovery
- Real Food, Fake Food: Why You Don't Know What You're Eating and What You Can Do about It
- The Diabetes Code: Prevent and Reverse Type 2 Diabetes Naturally
- The Longevity Solution: Rediscovering Centuries-Old Secrets to a Healthy, Long Life
- The Obesity Code: Unlocking the Secrets of Weight Loss
- The Dorito Effect: Mark Schatzker
- The Hacking of the American Mind: Robert H Lustig
- The Way We Eat Now: Bee Wilson
- Ultra Processed People: Chris van Tulleken

Conversation History:
{history}

Context:
{context}

Health Conversation History:
{nutri}

General Conversation History:
{gen}

Expected Response Type:
{expected}

---
IMPORTANT POINTS:
- Always refer to conversation history to know what the question asked is talking about.
- Respond in a human like manner.
- If expected response type is health, then only refer to context, conversation history, health conversation history, rephrased question, question, to answer accurately and precisely.
- If expected response type is general, then only refer to the question, conversation history, general conversation history, rephrased question and general conversation history to answer.
- For general expected responses, answer only what the questions asked says, nothing more than that.

Question asked:
- {question}
Rephrased question
- {q_rephrase}
"""

# In-memory conversation history storage
conversation_history = []
health_conversation_history = []
general_conversation_history = []
current_reponse = ""

def classify_question(query_text, model):
    last_convo =""
    if conversation_history:
        last_convo = " ".join([conversation_history[-1], conversation_history[-2]])
    else:
        last_convo = "the conversation just started"
    # Use your ChatGroq model to classify whether the question is health or general
    response = model.invoke(f"""
                            Determine whether the current conversation text is linked to 'general' or 'health' conversation based on the following information:
                            the current conversation text: {query_text}.
                            the last relevant conversation: {last_convo}.
                            'health' means the text talks about anything related to human body, health, biology, chemistry, biochemistry, nutrition, food, or any science related to these fields.
                            'general' means the text does not relate to health or the specified sciences and 
                            can be answered independently without context, 
                            general information also includes information about the chatbot, 
                            such as its knowledge base, books it was taught on, or its details.
                            Your answer should be based on both conversations because the current conversation may be dependent on the previous conversation.
                            IMPORTANT POINT:
                            - If you feel that the answer is 'health', then your final response should '8512'.
                            - If you feel that the answer is 'general', then your final response should be '6781'.
                            - By no means include both numbers in your response, you response is used to detect which number is present, which is important for further evaluation.
                            - Answer should be a number, which is either '8512' or '6781', but do not include both in the response.
                            """)
    # Extracting the response content
    classification_result = response.content.strip()

    print(f"\n\nTemp: {classification_result}\n\n")
    
    global current_response
    
    if "8512" in classification_result.lower():
        print("\nHealth Term Detected\n")
        current_response = "health"
        return "health"
    elif "6781" in classification_result.lower():
        print("\nGeneral Term Detected\n")
        current_response = "general"
        return "general"
    else:
        print("\nNo Term Detected, resorting to health\n")
        current_response = "health"
        return "health"

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
        if category == "health":
            health_conversation_history.append(f"Client: {query_text}")
        else:
            general_conversation_history.append(f"Client: {query_text}")

def query_rag(query_text: str, category: str, model):
    # Prepare the DB
    embedding_function = CohereEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Select the appropriate conversation history based on category
    if category == "health":
        current_history = health_conversation_history
    else:
        current_history = general_conversation_history
    
    # Rephrase the query if it is health relevant
    if category == "health":
        if health_conversation_history:
            last_health_convo = " ".join([health_conversation_history[-1], health_conversation_history[-2]])
            rephrased_query = f"Previous health conversation text and reply here: {last_health_convo}; Current question: {query_text}"
        else:
            rephrased_query = query_text
        print(f"\n\nRephrasing: {rephrased_query}\n\n")
    else:
        rephrased_query = query_text
    
    # Search the DB with the rephrased query
    results = db.similarity_search_with_score(rephrased_query, k=4)

    # Construct the context text only if category is health
    if category == "health":
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    else:
        context_text = ""
    
    # Construct the conversation history text
    history_text = "\n".join(conversation_history + current_history)

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the input questions for the prompt
    formatted_prompt = prompt_template.format(
        context=context_text,
        history=conversation_history,
        nutri=health_conversation_history,
        gen=general_conversation_history,
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