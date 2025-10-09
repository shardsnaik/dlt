# from llama_cpp import Llama
# import os

# # Path to GGUF model file
# MODEL_PATH = r"C:\Users\Public\Rag_based_chatbot\Medical_bot\model\medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf"
# # unsloth/gemma-3-4b-it-GGUF
# # Use all available CPU threads
# NUM_THREADS = os.cpu_count()

# # Load the model
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=100000,       # Context size
#     n_threads=NUM_THREADS,
#     n_batch=512,      # Increases throughput
#     verbose=False
# )

# # Med-Gemma specific prompt formatting
# prompt = """<start_of_turn>user
# Explain diarrhea and how can I take precautions myself<end_of_turn>
# <start_of_turn>assistant
# """

# response = llm(
#     prompt,
#     max_tokens=256,
#     temperature=0.7,
#     top_p=0.9,
#     stop=["<end_of_turn>"]  # Important for stopping generation
# )

# print(response["choices"][0]["text"].strip())


# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"   # keep TF/keras out
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster downloads if hf_transfer installed
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import torch

# BASE_MODEL = "google/medgemma-4b-it"

# ADAPTER_DIR = "C:\\Users\\Public\\Rag_based_chatbot\\Medical_bot\\Models\\lora_adapters"

# # peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
# # print("Adapter was trained on:", peft_cfg.base_model_name_or_path)

# # Load base (prefer fp16)
# base = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float32,
#     device_map="cpu",
#     low_cpu_mem_usage=True
# )
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# lora = PeftModel.from_pretrained(base, ADAPTER_DIR)
# merged = lora.merge_and_unload()

# # Save merged in safetensors (recommended)
# merged.save_pretrained('C:\\Users\\Public\\Rag_based_chatbot\\Medical_bot\Models\\medgemma_merged', safe_serialization=True)
# tokenizer.save_pretrained('C:\\Users\\Public\\Rag_based_chatbot\\Medical_bot\Models\\medgemma_merged')

# print("Merged model saved to:", 'C:\\Users\\Public\\Rag_based_chatbot\\Medical_bot\Models\\medgemma_merged')


#########################################################
# Trials of Lanchain chain function with and without
###################################################

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Grok API endpoint
    api_key=os.getenv('api')
)
mongodb_client = AsyncIOMotorClient(os.getenv("mongodb_uri"))
mongodb = mongodb_client["chatbot_db"]


def ask_grok_direct(question: str):
    # Step 1: Build prompt manually
    system_prompt = "You are a helpful assistant who explains things simply."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Step 2: Call Grok
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )

    # Step 3: Extract and clean
    reply = response.choices[0].message.content.strip()
    return reply

async def run():
    print("💬 User: What is LangChain?")
    answer = ask_grok_direct("What is LangChain?")
    print("🤖 Grok:", answer)
    data_chart_to_save = {
            'user query': 'What is LangChain?',
            'responce': answer
            }
        # Insert into MongoDB
        
    result = await mongodb["chat_logs"].insert_one(data_chart_to_save)
    print("✅ Saved to DB with ID:", result.inserted_id)

import asyncio
if __name__ == "__main__":
    asyncio.run(run())
    





# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnableLambda
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
# from langchain.schema import BaseOutputParser
# from openai import OpenAI
# import os

# # -----------------------------
# # 1️⃣ Initialize Grok client
# # -----------------------------

# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",  # Grok API endpoint
#     api_key=os.getenv('pio))
# # )


# # -----------------------------
# # 2️⃣ Define Chat Prompt Template
# # -----------------------------
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a friendly AI medical assistant."),
#     MessagesPlaceholder(variable_name="chat_history"),   # Add memory placeholder
#     ("human", "{input}"),
# ])


# # -----------------------------
# # 3️⃣ Define a function that calls Grok
# # -----------------------------
# def call_grok(prompt_messages):
#     """Convert LangChain messages -> Grok messages and call API."""
#     print(prompt_messages)
#     openai_messages = []
#     role = 'assistant' if 'HumanMessage' in prompt_messages else 'user'
#     content = prompt_messages

        
        
#     openai_messages.append({"role": role, "content": content})

#     response = client.chat.completions.create(
#         model="grok-beta",
#         messages=openai_messages,
#         temperature=0.7,
#         max_tokens=200
#     )

#     return response.choices[0].message.content


# # Wrap as a RunnableLambda (so it fits into LangChain’s pipeline)
# llm = RunnableLambda(call_grok)


# # -----------------------------
# # 4️⃣ Define an Output Parser
# # -----------------------------
# class SimpleOutputParser(BaseOutputParser):
#     def parse(self, text: str) -> str:
#         return text.strip()


# # -----------------------------
# # 5️⃣ Combine them into a pipeline
# # -----------------------------
# chain = prompt | llm | SimpleOutputParser()


# # -----------------------------
# # 6️⃣ Setup Chat History (Memory)
# # -----------------------------
# # Each session_id gets its own message history
# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


# # -----------------------------
# # 7️⃣ Wrap chain with message history support
# # -----------------------------
# conversation_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history"
# )


# # -----------------------------
# # 8️⃣ Chat function — keeps context
# # -----------------------------
# def chat(user_input: str, session_id="session_1"):
#     response = conversation_chain.invoke(
#         {"input": user_input},
#         config={"configurable": {"session_id": session_id}}
#     )
#     return response


# # -----------------------------
# # 9️⃣ Example Chat Usage
# # -----------------------------
# if __name__ == "__main__":
#     print("🩺 Grok Medical Chatbot (with memory)")
#     while True:
#         user_input = input("\n👤 You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         reply = chat(user_input)
#         print(f"🤖 Grok: {reply}")
        
