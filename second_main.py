from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from importlib.metadata import version

load_dotenv()



def demo_basic_chain():
    """Demonstrate the basic chain using LCEL and runnables"""

    #Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )

    model = ChatGroq(model = "llama-3.1-8b-instant")
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question" : "Compare langchain and Langgraph"})
    print("Response: \n", result)

    return chain


def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template(
        "Translate the text to French: {text}"
    )

    model = ChatGroq(model = "llama-3.1-8b-instant")
    
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - Run with multiple inputs
    inputs = [
        {"text" : "Hello, How are you?"},
        {"text" : "Whats your profession?"},
        {"text" : "Im a CS major student residing in Germany."}
    ]

    result = chain.batch(inputs)

    for text in zip(inputs, result):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template(
        "Create a short Shakespare like novel on {topic}. Try to mimic the style and the language of shakespare."
    )
    model = ChatGroq(model = "llama-3.1-8b-instant")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming Output: ")
    for chunk in chain.stream({"topic" : "Larger part of society shifting from Ownership to Sharing. Few own islands yet many do not have a place to call a home. Larger reason is coorporate greed leading to financial inequality. This isn't so happy thing, yet institutions celebrate it in the name of You'll be happy"}):
        print(chunk, end = "", flush= True)
    print()


def demo_schema_inspection():
    """Demonstrate input/output schema inscpection"""

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text: {text}"
    )
    model = ChatGroq(model = "llama-3.1-8b-instant")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspecting Output and input schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print("Input Schema: \n", input_schema)
    print("Output Schema: \n", output_schema)

    return chain



def exercise_first_chain():

    """
    Exercise: Create a chain that :
    1: Takes a product name and target audience.
    2: Generates a marketing tagline.
    3: Returns just the tagline as string.


    Test with product : AI Course and Audience : Developer 
    """


    prompt = ChatPromptTemplate.from_template(
        "You are a marketing expert, Generate a Marketing tagline for the product {product} with {audience} as the target audience. I want straight answer, just a single marketing tagline."
    )
    model = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature=0.8,
        max_tokens=1000,
        timeout=30,
        max_retries=3
        )
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product" : "AI Course", "audience" : "Developer"})

    print("Generated Tagline: \n ", result)

    return chain


# MAIN Function

# out = demo_basic_chain()
# print(out)

# demo_batch_execution()
# demo_streaming()

# demo_schema_inspection()
exercise_first_chain()