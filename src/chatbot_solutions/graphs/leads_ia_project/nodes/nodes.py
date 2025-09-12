from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from ..crew_ai_agents.agents_schema import AgentState
from ..prompts.leads_prompt import system_prompt
from ..tools.tools import ALL_TOOLS
from ..llm.llm import llm

tool_executor = ToolNode(ALL_TOOLS)

# Vincula as ferramentas ao LLM para habilitar chamadas estruturadas de ferramentas
model = llm.bind_tools(ALL_TOOLS)

def call_model(state: AgentState) -> dict:  # Retorna um dict para atualizar o estado
    previous_messages = list(state["messages"])
    all_messages = [system_prompt] + previous_messages
    try:
        response = model.invoke(all_messages)
        # Retorna a lista de mensagens atualizada com a resposta da IA
        return {"messages": previous_messages + [response]} 
    except Exception as e:
        return {"messages": previous_messages + [AIMessage(content=f"[ERRO] Falha no OpenAI: {str(e)}")]}

