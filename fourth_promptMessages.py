from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate



# Multi Message Templates

# model = init_chat_model(
#     model = "llama-3.1-8b-instant",
#     model_provider= "groq"
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system", "You are a helpful assistant that translates {input_language} into {output_language}"
#         ),
#         (
#             "human" , "Translate the following text : {text}"
#         )
#     ]
# )

# messages = prompt.format_messages(
#     input_language = "English", output_language = "French", text = "I love programming."
# )

# response = model.invoke(messages)

# print(response.content)





# Fewshot Example
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input" : "Happy", "output" : "Sad"},
    {"input" : "tall", "output" : "short"}
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("assistant", "{output}"),
])

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "Give the opposite of each words."),
        fewshot_prompt,
        ("human", "{input}")
    ]
)

model = init_chat_model(
        model="llama-3.1-8b-instant",
    model_provider="groq")
response= model.invoke(final_prompt.format_messages(input = "happy"))


print(response.content)