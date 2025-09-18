from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from ..crew_ai_agents.agents_schema import AgentState
from ..prompts.leads_prompt import system_prompt
from ..tools.tools import ALL_TOOLS
from ..llm.llm import llm
from .utils import extract_name_and_args, execute_tool_locally, format_observations_for_model
from typing import List

tool_executor = ToolNode(ALL_TOOLS)
model_with_tools = llm.bind_tools(ALL_TOOLS)

def call_model(state: AgentState) -> dict:
    previous_messages: List[BaseMessage] = list(state["messages"])

    # Passo 1: modelo com ferramentas para planejar
    planning_messages = [system_prompt] + previous_messages
    first_response: AIMessage = model_with_tools.invoke(planning_messages)

    # Se não houver tool_calls, devolve direto
    tool_calls = getattr(first_response, "tool_calls", None)
    if not tool_calls:
        return {"messages": previous_messages + [first_response]}

    # Passo 2: executa tool_calls internamente (sem ToolNode, sem role:"tool")
    executed = []
    for tc in tool_calls:
        name, args = extract_name_and_args(tc)
        result = execute_tool_locally(name, args)
        executed.append({"tool": name, "args": args, "result": result})

    # Passo 3: segunda chamada ao modelo SEM ferramentas, entregando observações para redigir a resposta final
    observations_text = format_observations_for_model(executed)

    final_messages = (
        [system_prompt] +
        previous_messages +  # histórico original (sem inserir nenhuma tool message)
        [
            # Dica clara ao modelo: use os resultados para responder ao usuário
            HumanMessage(content=(
                "Use os resultados abaixo para formular a resposta final ao usuário, "
                "sem mencionar ferramentas ou chamadas técnicas.\n\n"
                f"{observations_text}"
            ))
        ]
    )

    final_response: AIMessage = llm.invoke(final_messages)
    return {"messages": previous_messages + [final_response]}

