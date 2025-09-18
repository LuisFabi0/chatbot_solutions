import json
from typing import Any, Dict, List
from ..tools.tools import ALL_TOOLS

# Dicionário global de ferramentas por nome
_TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in ALL_TOOLS}


def extract_name_and_args(tc: Any) -> tuple[str, Dict[str, Any]]:
    """
    Extrai (name, args) de um item de tool_call, cobrindo formatos comuns do LangChain/OpenAI.
    """
    # Formato LangChain típico: {"id": "...", "name": "foo", "args": {...}}
    name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)

    args = None
    if isinstance(tc, dict):
        args = tc.get("args")
        # Formato OpenAI function-calling: {"function": {"name": "...", "arguments": "json"}}
        if not name and "function" in tc:
            name = tc["function"].get("name")
            arguments = tc["function"].get("arguments")
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except Exception:
                    args = {"input": arguments}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"input": args}
    else:
        # objetos com atributos
        args = getattr(tc, "args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"input": args}

    if args is None:
        args = {}

    return name or "", args


def execute_tool_locally(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """
    Executa a ferramenta localmente, sem enviar role:'tool' para a OpenAI.
    Não disparamos webhook de tool message; apenas chamamos a função Python correspondente.
    """
    tool = _TOOLS_BY_NAME.get(tool_name)
    if tool is None:
        return {"error": f"Ferramenta '{tool_name}' não encontrada."}

    try:
        # Preferimos .invoke(args); se não existir, tentamos .run ou __call__
        if hasattr(tool, "invoke"):
            return tool.invoke(tool_args)
        if hasattr(tool, "run"):
            return tool.run(tool_args)
        return tool(tool_args)  # pode levantar exceção se não suportar dict
    except Exception as e:
        return {"error": f"Falha ao executar '{tool_name}': {str(e)}"}


def format_observations_for_model(results: List[Dict[str, Any]]) -> str:
    """
    Concatena resultados das ferramentas em um texto legível para alimentar o segundo passo do modelo.
    """
    parts = []
    for item in results:
        tool = item.get("tool")
        output = item.get("result")
        try:
            # Normaliza para string
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
        except Exception:
            output = str(output)
        parts.append(f"- {tool}: {output}")
    return "Resultados das operações solicitadas:\n" + "\n".join(parts)
