from __future__ import annotations
import os, re
from pathlib import Path
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
            "CNPJ": LEAD_STATE.get("cnpj"),
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
            # Garante que campos essenciais sempre apareçam como "Não informado" se vazios
            if k in ["CNPJ", "Interesse", "Tamanho do Time"] and not v:
                lines.append(f"{k}: Não informado\n")
            else:
                lines.append(f"{k}: {v or 'Não informado'}\n")
        lines.append("\nExplicação:\n")
        lines.append(explanation)

        # Usa o nome do cliente como nome do arquivo, removendo caracteres especiais
        nome_cliente = (LEAD_STATE.get('nomeLead') or 'cliente_desconhecido').strip()
        # Remove caracteres especiais e espaços para criar um nome de arquivo válido
        nome_arquivo = re.sub(r'[^\w\s-]', '', nome_cliente).strip()
        nome_arquivo = re.sub(r'[-\s]+', '_', nome_arquivo)
        filename = f"{nome_arquivo}_lead_result.txt"

        # Encontra a pasta storage na raiz do projeto
        try:
            # Sobe na hierarquia até encontrar a pasta storage na raiz
            current_path = Path(__file__).resolve()
            project_root = None
            
            # Procura pela pasta storage subindo na hierarquia
            for parent in current_path.parents:
                storage_path = parent / "storage"
                # Verifica se é a pasta storage da raiz (não a do projeto leads)
                if storage_path.exists() and (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            
            if not project_root:
                # Fallback: usa a pasta storage relativa ao diretório atual de execução
                project_root = Path.cwd()
                
        except Exception:
            project_root = Path.cwd()

        output_dir = project_root / "storage"
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        # Envia o arquivo por email automaticamente
        try:
            from ..email.email_sender import enviar_arquivo_txt_por_email
            resultado_email = enviar_arquivo_txt_por_email(str(file_path))
            if resultado_email:
                print(f"✅ Email enviado automaticamente com o arquivo: {file_path}")
            else:
                print(f"❌ Falha no envio do email para o arquivo: {file_path}")
        except ImportError as e:
            print(f"❌ Erro de importação do email_sender: {e}")
        except Exception as e:
            print(f"❌ Erro inesperado ao enviar email: {e}")
            import traceback
            traceback.print_exc()
        
        # Atualiza o dashboard HTML automaticamente
        try:
            from utils.dashboard_updater import update_dashboard
            update_dashboard()
        except Exception:
            pass
            
    except Exception:
        pass

    return _safe_json({"ok": True, "acao": "registrar_lead", "status_lead": status, "resultado_rd": result})


@tool
def salvar_dado_lead(campo: str, valor: str) -> str:
    """
    Atualiza LEAD_STATE[campo] e tenta sincronizar com RD (create/update) quando houver e-mail.
    """
    campo = (campo or "").strip()
    valor_normalizado = (valor or "").strip() or None

    if not campo:
        return _safe_json({
            "ok": False,
            "erro": "campo inválido",
            "lead": LEAD_STATE
        })

    campo_normalizado = re.sub(r"[^a-z0-9]", "", campo.lower())
    cnpj_aliases = {
        "cnpj",
        "cnpjcliente",
        "cnpjempresa",
        "cnpjcpf",
        "cpfcnpj",
        "cpfcnpjcliente",
        "document",
        "documento",
    }

    if campo_normalizado in cnpj_aliases:
        LEAD_STATE["cnpj"] = valor_normalizado
        # Append CNPJ to cnpjs.txt (one per line)
        try:
            cnpj_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "txt_files",
                "cnpjs.txt"
            )
            with open(cnpj_file_path, "a", encoding="utf-8") as f:
                if valor_normalizado:
                    f.write(f"{valor_normalizado}\n")
        except Exception:
            pass
        if campo != "cnpj":
            LEAD_STATE[campo] = valor_normalizado
    else:
        LEAD_STATE[campo] = valor_normalizado

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

