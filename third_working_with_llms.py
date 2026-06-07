"""
Working with LLMs in LangChain V.1
Multiple providers, configuartions, streaming and cost optimization
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()



def demo_init_chat_model():
    prompt = ChatPromptTemplate.from_template(
       "What's the capital of {Country}. Answer in one word."
    )

    # This is agnostic way to create a model
    model = init_chat_model(
        model = "llama-3.1-8b-instant",
        model_provider= "groq",
        temperature = 0.7,
        streaming = True,
        max_retries = 3
    )

    parser = StrOutputParser()

    chain = prompt | model | parser


    response = chain.invoke({"Country" : "Bharat"})
    print(response)

    return chain


def demo_messages():
    model = init_chat_model(
        model= "llama-3.1-8b-instant",
        model_provider="groq"
    )
    
    query = input("Ask Pirate Something: ")

    # Using messages object => Gives more control over roles.
    messages = [
            # 👉 Sets rules / behavior for the AI
            # 👉 “How should the AI behave?”
            SystemMessage(
                content="You are a pirate. Act like one while answering."
            ),
            # 👉 Input from the user
            # 👉 “What the user is asking”
            HumanMessage(
                content= query
            )
        ]
    
    for chunk in model.stream(messages):
        print (chunk.content, end="", flush=True)
    print()



# # Multi turn conversation using message object
def demo_message():
    model = ChatGroq(model = "llama-3.1-8b-instant")

    # Using messages object => Gives more control over roles.
    messages = [
        SystemMessage(content = "You are a pirate. Always answer like one."),
        HumanMessage(content = "Whats your opinion on The vessel of life?"),
        # 👉 Output from the model
        # 👉 “What the AI replied”
        # AIMessage()
    ]

    response = model.invoke(messages)
    print(f"Response from the pirate: {response.content}")


    # Multi turn conversation using message object
    messages.append(response) #add model's response to the conversation
    messages.append(HumanMessage(content="Whats your opinion on the cold nights when theres hardly any food left on the ship?"))


    print("\n Multi turn conversation:")
    response = model.invoke(messages)
    print(f"Follow up question from the pirate: {response.content}")


# Multi turn conversation exercise using System Messages and Human Messages.
def exercise_multi_turn_newsAnchor():
    model = init_chat_model(
        model = "llama-3.1-8b-instant",
        model_provider="groq"
    )

    query = input("Ask Arnab Goswami: ")

    messages = [
        SystemMessage(content="You are an English new anchor, answer like one. Keep the answers within 50 words."),
        HumanMessage(content=query)
    ]
    

    while query != "STOP":

        larger_context = ""

        for chunk in model.stream(messages):
            print(chunk.content, end = "", flush=False)
            larger_context += chunk.content
        print()

        query = input("Ask next question: \n")
        messages.append(larger_context) # adds model's response to the conversation
        messages.append(HumanMessage(content=query))
  

# Main Function
# demo_init_chat_model()
# demo_messages()
# demo_message()
exercise_multi_turn_newsAnchor()
