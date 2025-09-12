import os, requests, json
from typing import Dict, Any, Optional
from requests.exceptions import Timeout, HTTPError, RequestException


RD_TOKEN = (os.getenv("RD_TOKEN") or "").strip()
BASE_RD = "https://crm.rdstation.com/api/v1/contacts"
RD_CUSTOM_FIELDS_BASE = "https://crm.rdstation.com/api/v1/custom_fields"


def _rd_headers() -> Dict[str, str]:
    return {"accept": "application/json", "content-type": "application/json"}

def _rd_listar_custom_fields() -> list[dict]:
    if not RD_TOKEN:
        return []
    url = f"{RD_CUSTOM_FIELDS_BASE}?token={RD_TOKEN}"
    try:
        r = requests.get(url, headers=_rd_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "custom_fields" in data:
            return data["custom_fields"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _rd_criar_custom_field(label: str, field_type: str = "text") -> Optional[dict]:
    if not RD_TOKEN:
        return None
    url = f"{RD_CUSTOM_FIELDS_BASE}?token={RD_TOKEN}"
    payload = {"custom_field": {"label": label, "type": field_type}}
    try:
        r = requests.post(url, headers=_rd_headers(), data=json.dumps(payload), timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("custom_field", data)
    except Exception:
        return None


def _ensure_custom_fields_exist(required: Dict[str, str]) -> Dict[str, str]:
    """
    required: dict lógico->label. Retorna dict lógico->key (do RD).
    Ex.: {"produtos_interesse": "Produtos de interesse"}
    """
    keys: Dict[str, str] = {}
    existing = _rd_listar_custom_fields()
    for logical, label in required.items():
        found = next((f for f in existing if f.get("label", "").strip().lower() == label.lower()), None)
        if found and found.get("key"):
            keys[logical] = found["key"]
        else:
            created = _rd_criar_custom_field(label)
            if created and created.get("key"):
                keys[logical] = created["key"]
    return keys

def _rd_listar_contatos(emailLead: str) -> Dict[str, Any]:
    """
    Lista contatos por e-mail.
    Retorna: {"ok":bool, "url":str, "result":Any, "idContato":str|None, "erro":str?}
    """
    email = (emailLead or "").strip()
    if not email:
        return {"ok": False, "erro": "emailLead ausente"}
    if not RD_TOKEN:
        return {"ok": False, "erro": "RD_TOKEN ausente"}

    url = f"{BASE_RD}?email={requests.utils.quote(email)}&token={RD_TOKEN}"
    try:
        resp = requests.get(url, headers=_rd_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Tenta extrair o idContato em formatos comuns
        found_id = None
        if isinstance(data, dict):
            if isinstance(data.get("contacts"), list) and data["contacts"]:
                found_id = data["contacts"][0].get("id") or data["contacts"][0].get("_id")
            else:
                found_id = data.get("id") or data.get("_id")
        elif isinstance(data, list) and data:
            found_id = data[0].get("id") or data[0].get("_id")

        return {"ok": True, "url": url, "result": data, "idContato": str(found_id) if found_id else None}
    except (Timeout, HTTPError, RequestException) as e:
        return {"ok": False, "erro": str(e), "url": url}


def _rd_criar_contato(
    nomeLead: str,
    emailLead: str,
    numeroCliente: str = "",
    cargoCliente: str = "",
    custom_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Cria contato.
    Retorna: {"ok":bool, "url":str, "payload":dict, "result":Any, "idContato":str|None, "erro":str?}
    """
    if not RD_TOKEN:
        return {"ok": False, "erro": "RD_TOKEN ausente"}
    email = (emailLead or "").strip()
    if not email:
        return {"ok": False, "erro": "emailLead ausente para criação"}

    payload = {
        "contact": {
            "name": (nomeLead or None),
            "emails": [{"email": email}],
            "phones": [{"type": "home", "phone": (numeroCliente or "")}] if numeroCliente else [],
            "title": (cargoCliente or None),
            "custom_fields": custom_fields or None,
        }
    }
    url = f"{BASE_RD}?token={RD_TOKEN}"
    try:
        resp = requests.post(url, headers=_rd_headers(), data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        new_id = data.get("id") or data.get("_id")
        if not new_id and isinstance(data, dict) and "contact" in data:
            new_id = data["contact"].get("id") or data["contact"].get("_id")

        return {
            "ok": True,
            "url": url,
            "payload": payload,
            "result": data,
            "idContato": str(new_id) if new_id else None,
        }
    except (Timeout, HTTPError, RequestException) as e:
        return {"ok": False, "erro": str(e), "url": url, "payload": payload}


def _rd_exibir_contato(idContato: str) -> Dict[str, Any]:
    """
    Exibe contato por ID.
    Retorna: {"ok":bool, "url":str, "result":Any, "erro":str?}
    """
    if not RD_TOKEN:
        return {"ok": False, "erro": "RD_TOKEN ausente"}
    cid = (idContato or "").strip()
    if not cid:
        return {"ok": False, "erro": "idContato ausente"}

    url = f"{BASE_RD}/{cid}?token={RD_TOKEN}"
    try:
        resp = requests.get(url, headers=_rd_headers(), timeout=15)
        resp.raise_for_status()
        return {"ok": True, "url": url, "result": resp.json()}
    except (Timeout, HTTPError, RequestException) as e:
        return {"ok": False, "erro": str(e), "url": url}


def _rd_atualizar_contato(
    idContato: str,
    tokenRD: str = "",
    nomeLead: str = "",
    emailLead: str = "",
    numeroCliente: str = "",
    cargoCliente: str = "",
    custom_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Atualiza contato por ID.
    Retorna: {"ok":bool, "url":str, "payload":dict, "result":Any, "erro":str?}
    """
    cid = (idContato or "").strip()
    if not cid:
        return {"ok": False, "erro": "idContato ausente"}

    token_use = (tokenRD or RD_TOKEN).strip()
    if not token_use:
        return {"ok": False, "erro": "RD_TOKEN ausente"}

    payload = {
        "name": (nomeLead or None),
        "title": (cargoCliente or None),
        "emails": [{"email": emailLead}] if emailLead else [],
        "phones": [{"type": "home", "phone": numeroCliente}] if numeroCliente else [],
        "custom_fields": custom_fields or None,
    }
    url = f"{BASE_RD}/{cid}?token={token_use}"
    try:
        resp = requests.put(url, headers=_rd_headers(), data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        return {"ok": True, "url": url, "payload": payload, "result": resp.json()}
    except (Timeout, HTTPError, RequestException) as e:
        return {"ok": False, "erro": str(e), "url": url, "payload": payload}
