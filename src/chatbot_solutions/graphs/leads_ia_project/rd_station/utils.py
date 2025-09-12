from typing import List, Dict, Any, Optional
import json
from .methods import _ensure_custom_fields_exist


# Essas informações são usadas pela CrewAI quando ela é acionada.
ROBBU_DOCS_CONTEXT = [
    {"name": "Carteiro Digital", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital"},
    {"name": "Webhook", "url": "https://docs.robbu.global/docs/center/webhook"},
    {"name": "Webchat", "url": "https://docs.robbu.global/docs/center/web-chat"},
    {"name": "Invênio", "url": "https://robbu.global/produtos/invenio-center/"},
    {"name": "Invenio live", "url": "https://robbu.global/produtos/invenio-live/"},
    {"name": "IDR Studio", "url": "https://robbu.global/produtos/idr-chatbot-studio/"},
    {"name": "Invenio Webchat", "url": "https://robbu.global/produtos/webchat/"},
    {"name": "Positus WhatsApp", "url": "https://robbu.global/produtos/whatsapp-studio-positus/"},
    {"name": "rFlow", "url": "https://robbu.global/produtos/rflow/"},
    {"name": "Campanha email", "url": "https://docs.robbu.global/docs/center/campanhas-de-email"},
    {"name": "Campanha whats app", "url": "https://docs.robbu.global/docs/center/campanhas-de-whatsapp"},
]

# ESTADO DO LEAD (memória de sessão)
LEAD_STATE: Dict[str, Optional[str]] = {
    "emailLead": None,
    "nomeLead": None,
    "siteEmpresa": None,
    "cargoCliente": None,
    "numeroCliente": None,
    "idContato": None,
    "status_lead": None,          # "quente" | "frio"
    "interesse_produto": None,    # Chatbot IA, Invênio, Carteiro Digital, Campanhas, Outros
    "tamanho_time": None,         # Número (string)
    "segmento": None,             # Segmento informado
}

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

_REQUIRED_CF = {
    "produtos_interesse": "Produtos de interesse",
    "qtd_funcionarios": "qtd de funcionarios",
}

def _build_custom_fields() -> Dict[str, Any]: # Esta função **depende da importação do _ensure_custom_fields_exist**
    keys = _ensure_custom_fields_exist(_REQUIRED_CF)
    cf: Dict[str, Any] = {}
    if LEAD_STATE.get("interesse_produto") and keys.get("produtos_interesse"):
        cf[keys["produtos_interesse"]] = LEAD_STATE["interesse_produto"]
    if LEAD_STATE.get("tamanho_time") and keys.get("qtd_funcionarios"):
        cf[keys["qtd_funcionarios"]] = LEAD_STATE["tamanho_time"]
    return cf
