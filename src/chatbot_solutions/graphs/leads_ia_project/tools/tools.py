from __future__ import annotations
import os, re
from typing import Any, Dict
from ..crew_ai_agents.agents_schema import TechnicalCrewExecutor
from langchain_core.tools import tool
from ..rd_station.utils import LEAD_STATE, _safe_json, _build_custom_fields 
from ..rd_station.methods import (
    _rd_listar_contatos,
    _rd_criar_contato,
    _rd_atualizar_contato,
)

@tool
def registrar_lead(status_lead: str) -> str:
    """
    Cria/atualiza contato no RD Station e define LEAD_STATE['status_lead'].
    Também gera um .txt de resumo do lead localmente (best effort).
    """
    status = (status_lead or "").strip().lower()
    if status not in {"quente", "frio", "desqualificado"}:
        return _safe_json({"ok": False, "erro": "status_lead inválido. Use 'quente', 'frio' ou 'desqualificado'."})

    email = (LEAD_STATE.get("emailLead") or "").strip()
    result: Dict[str, Any]

    custom_fields = _build_custom_fields() or None

    if LEAD_STATE.get("idContato"):
        result = _rd_atualizar_contato(
            idContato=LEAD_STATE["idContato"],
            nomeLead=LEAD_STATE.get("nomeLead") or "",
            emailLead=email,
            numeroCliente=LEAD_STATE.get("numeroCliente") or "",
            cargoCliente=LEAD_STATE.get("cargoCliente") or "",
            custom_fields=custom_fields,
        )
    else:
        created = _rd_criar_contato(
            nomeLead=LEAD_STATE.get("nomeLead") or "",
            emailLead=email,
            numeroCliente=LEAD_STATE.get("numeroCliente") or "",
            cargoCliente=LEAD_STATE.get("cargoCliente") or "",
            custom_fields=custom_fields,
        )
        # atualiza idContato no estado se veio do RD
        new_id = (created.get("idContato") or created.get("new_id"))
        if new_id:
            LEAD_STATE["idContato"] = str(new_id)
        result = created

    LEAD_STATE["status_lead"] = status

    # grava .txt local com resumo (opcional)
    try:
        lead_info = {
            "Nome": LEAD_STATE.get("nomeLead"),
            "Email": LEAD_STATE.get("emailLead"),
            "Site": LEAD_STATE.get("siteEmpresa"),
            "Cargo": LEAD_STATE.get("cargoCliente"),
            "Número": LEAD_STATE.get("numeroCliente"),
            "Interesse": LEAD_STATE.get("interesse_produto"),
            "Tamanho do Time": LEAD_STATE.get("tamanho_time"),
            "Segmento": LEAD_STATE.get("segmento"),
            "Status do Lead": LEAD_STATE.get("status_lead"),
        }
        if status == "quente":
            explanation = (
                "Este lead é considerado QUENTE porque atende aos critérios:\n"
                "- Empresa com mais de 5 funcionários\n"
                "- Possui site ativo\n"
                "- Tem interesse real em algum produto/case da Robbu\n"
            )
        elif status == "frio":
            explanation = (
                "Este lead é considerado FRIO porque não atende a todos os critérios necessários:\n"
                "- Empresa com mais de 5 funcionários\n"
                "- Possui site ativo\n"
                "- Tem interesse real em algum produto/case da Robbu\n"
            )
        else:
            explanation = "Status do lead é DESQUALIFICADO ou outro."

        lines = ["Resultado da Qualificação do Lead\n", "===============================\n"]
        for k, v in lead_info.items():
            lines.append(f"{k}: {v or 'Não informado'}\n")
        lines.append("\nExplicação:\n")
        lines.append(explanation)

        filename = f"lead_result_{(LEAD_STATE.get('emailLead') or 'unknown').replace('@','at').replace('.','')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass

    return _safe_json({"ok": True, "acao": "registrar_lead", "status_lead": status, "resultado_rd": result})


@tool
def salvar_dado_lead(campo: str, valor: str) -> str:
    """
    Atualiza LEAD_STATE[campo] e tenta sincronizar com RD (create/update) quando houver e-mail.
    """
    campo = (campo or "").strip()
    LEAD_STATE[campo] = (valor or "").strip() or None

    rd_result: Any = None
    try:
        email = (LEAD_STATE.get("emailLead") or "").strip()
        if email:
            # tenta localizar contato
            lookup = _rd_listar_contatos(emailLead=email)
            found_id = lookup.get("idContato")
            custom_fields = _build_custom_fields() or None

            if found_id:
                LEAD_STATE["idContato"] = str(found_id)
                rd_result = _rd_atualizar_contato(
                    idContato=str(found_id),
                    nomeLead=LEAD_STATE.get("nomeLead") or "",
                    emailLead=email,
                    numeroCliente=LEAD_STATE.get("numeroCliente") or "",
                    cargoCliente=LEAD_STATE.get("cargoCliente") or "",
                    custom_fields=custom_fields,
                )
            else:
                created = _rd_criar_contato(
                    nomeLead=LEAD_STATE.get("nomeLead") or "",
                    emailLead=email,
                    numeroCliente=LEAD_STATE.get("numeroCliente") or "",
                    cargoCliente=LEAD_STATE.get("cargoCliente") or "",
                    custom_fields=custom_fields,
                )
                new_id = (created.get("idContato") or created.get("new_id"))
                if new_id:
                    LEAD_STATE["idContato"] = str(new_id)
                rd_result = created
    except Exception as e:
        rd_result = f"[ERRO_RD_STATION:{str(e)}]"

    return _safe_json({
        "ok": True,
        "atualizado": {campo: LEAD_STATE[campo]},
        "lead": LEAD_STATE,
        "rd_station_result": rd_result
    })


@tool
def falar_com_atendente_humano() -> str:
    """Pede transferência para atendimento humano."""
    return "Transferência para atendimento humano solicitada."


@tool
def Get_finalizaCliente() -> str:
    """Finaliza o atendimento."""
    return _safe_json({"ok": True, "acao": "finalizar", "mensagem": "Atendimento finalizado. A Robbu agradece o contato!"})


@tool
def pesquisa_tecnica_avancada_robbu(query: str) -> str:
    """Pesquisa técnica baseada na documentação pública da Robbu."""
    if TechnicalCrewExecutor is None:
        return _safe_json({"ok": False, "erro": "TechnicalCrewExecutor não importado/configurado."})
    return TechnicalCrewExecutor().run(query)


@tool
def montar_requisicao(
    project: str,
    message: str,
    name: str = "",
    document: str = "",
    phone: str = "",
    email: str = "",
) -> str:
    """Monta payload com dados de contato (usa LEAD_STATE como fallback)."""
    try:
        _name = (name or LEAD_STATE.get("nomeLead") or "").strip()
        _phone = (phone or LEAD_STATE.get("numeroCliente") or "").strip()
        _email = (email or LEAD_STATE.get("emailLead") or "").strip()
        payload = {
            "message": message,
            "project": project,
            "contact": {
                "name": _name or None,
                "document": document or None,
                "channel": {"phone": _phone or None, "email": _email or None},
            },
        }
        return _safe_json({"ok": True, "payload": payload})
    except Exception as e:
        return _safe_json({"ok": False, "erro": str(e)})


@tool
def coleta_leads() -> str:
    """Sinaliza para continuar a coleta (uma pergunta por vez)."""
    return _safe_json({"ok": True, "acao": "coleta_leads", "lead": LEAD_STATE})


# Para plugar direto no LangGraph:
ALL_TOOLS = [
    pesquisa_tecnica_avancada_robbu,
    falar_com_atendente_humano,
    salvar_dado_lead,
    coleta_leads,
    registrar_lead,
    Get_finalizaCliente,
    montar_requisicao,
]

