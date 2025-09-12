import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_LEADS_IA_API_KEY = os.getenv("OPENAI_LEADS_IA_API_KEY")
if not OPENAI_LEADS_IA_API_KEY:
    raise ValueError("Chave da API da OpenAI não encontrada. Verifique seu arquivo .env")
 
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=OPENAI_LEADS_IA_API_KEY,
    temperature=0.8
)

