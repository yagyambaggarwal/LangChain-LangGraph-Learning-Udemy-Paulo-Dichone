from dotenv import load_dotenv
from importlib.metadata import version
import os
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic

def usingGoogle(prompt):

    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash"
    )
    response = llm.invoke(prompt)

    return response.content

def usingGroq(prompt):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )
    response = llm.invoke(prompt)
    return response.content

def main():
    # print("Langchain_core Version:", version("langchain"))
    # print("Langgraph Version:",version("langgraph"))


    responseFromGoogle = usingGoogle("Tell me a bedtime story of 10 just 10 words.")
    print(responseFromGoogle)

    responseFromGroq = usingGroq("Tell me the best stoic advice to be heard at 3 am in chilling cold night empty stomach.")
    print(responseFromGroq)


if __name__ == "__main__":
    main()
