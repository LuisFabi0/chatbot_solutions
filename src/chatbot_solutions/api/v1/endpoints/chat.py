from typing import List, Optional, Any

from zoneinfo import ZoneInfo
from datetime import datetime

from fastapi import APIRouter, status, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict

from models.usuario_model import UsuarioModel
from schemas.usuario_schema import MessageRequestSchema, UsuarioSchema, MessageResponseSchema, Contact, Channel
from core.deps import get_session
from graphs.graph_yamaha import langgraph_app
from graphs.help_desk_graph import APP
from graphs.agent_graph_leads import agent_graph_leads
from graphs.leads_ia_project.graph import leads_ia_graph
from webhook_calls import trigger_webhook_message, trigger_webhook_tool_call

br_tz = ZoneInfo("America/Sao_Paulo")

router = APIRouter()

@router.post('', response_model=MessageResponseSchema, status_code=status.HTTP_202_ACCEPTED)
async def post_chat(usuario: MessageRequestSchema, db: AsyncSession = Depends(get_session)):

    if not usuario.message:
        raise HTTPException(detail='A mensagem não pode estar em branco.', status_code=status.HTTP_400_BAD_REQUEST)

    messages = []
    previous_message = []

    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == usuario.contact.channel.phone,
                                            UsuarioModel.project == usuario.contact.project,
                                            UsuarioModel.protocol == usuario.contact.protocol)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()

        input_message = HumanMessage(content=usuario.message,metadata={"timestamp": datetime.now(br_tz).isoformat()})

        if not usuario_db:
            novo_usuario = UsuarioModel(
                protocol=usuario.contact.protocol,
                project=usuario.contact.project,
                nome=usuario.contact.name,
                document=usuario.contact.document,
                phone=usuario.contact.channel.phone,
                email=usuario.contact.channel.email,
                messages=messages_to_dict([input_message]),
                processing= True
            )
            session.add(novo_usuario)
            await session.commit()
            await session.refresh(novo_usuario)
            usuario_db = novo_usuario

        else:
            if usuario_db.processing == True:
                raise HTTPException(detail='Uma mensagem está sendo processada, não é possível adicionar mais mensagens', status_code=status.HTTP_406_NOT_ACCEPTABLE)
            previous_message = messages_from_dict(usuario_db.messages)
            previous_message.append(input_message)
            usuario_db.messages = messages_to_dict(previous_message)
            await session.commit()
            await session.refresh(usuario_db)

    messages = messages_from_dict(usuario_db.messages)

    contato = Contact(name=usuario_db.nome,
                      document=usuario_db.document,
                      project = usuario_db.project,
                      protocol = usuario_db.protocol,
                      channel=Channel(phone=usuario_db.phone,
                                      email=usuario_db.email))
    webhook_url = usuario.webhook_url


    state = {"messages": messages,
             "last_ai_message": None,
             "last_human_message":[HumanMessage(content=usuario.message)]
            }

    if usuario.contact.project == "Yamaha Cobrança IA":
        async for step in langgraph_app.astream(state, stream_mode="values"):
            final_state = step
    if usuario.contact.project == "HelpDesk IA":
        async for step in APP.astream(state, stream_mode="values"):
            final_state = step
    if usuario.contact.project == "Qualificador Leads IA":
        async for step in agent_graph_leads.astream(state, stream_mode="values"):
            final_state = step
    if usuario.contact.project == "Qualificador Leads IA2":
        async for step in leads_ia_graph.astream(state, stream_mode="values"):
            final_state = step

    last_ai_message = messages_to_dict(final_state["messages"])[-1]

    tool_calls = last_ai_message.get("data", {}).get("tool_calls", [])
    content = last_ai_message.get("data", {}).get("content")
    if tool_calls:
        await trigger_webhook_tool_call(contact= contato, tools= tool_calls, webhook_url = webhook_url)
    else:
        await trigger_webhook_message(contact= contato, message= content, webhook_url = webhook_url)

    response: MessageResponseSchema = MessageResponseSchema(data=final_state["messages"][-1].content, contact= contato)
    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == usuario.contact.channel.phone,
                                            UsuarioModel.project == usuario.contact.project,
                                            UsuarioModel.protocol == usuario.contact.protocol)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()

        if usuario_db:
            usuario_db.processing = False
            usuario_db.messages = messages_to_dict(final_state["messages"])
            await session.commit()
            await session.refresh(usuario_db)

    return response
