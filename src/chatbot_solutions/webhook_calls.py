import uuid
import httpx
from schemas.usuario_schema import Contact
import logging


logger = logging.getLogger(__name__)

async def trigger_webhook_tool_call(contact: Contact, tools: dict, webhook_url: str):
    payload = {
    "data": tools,
    "contact": contact.dict()
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Webhook Tool Call enviado com sucesso. Status: %s", response.status_code)
    except httpx.HTTPStatusError as e:
        logger.error("Erro HTTP ao enviar webhook Tool Call: %s - %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Erro de requisição ao enviar webhook Tool Call: %s", str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao enviar webhook Tool Call: %s", str(e))


async def trigger_webhook_message(contact: Contact, message: str, webhook_url: str):
    payload = {
    "data": message.replace('\n', '<br>') if isinstance(message, str) else message,
    "contact": contact.dict()
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Webhook Message enviado com sucesso. Status: %s", response.status_code)
    except httpx.HTTPStatusError as e:
        logger.error("Erro HTTP ao enviar webhook Message: %s - %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Erro de requisição ao enviar webhook Message: %s", str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao enviar webhook Message: %s", str(e))
