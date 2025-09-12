from typing import Optional
from typing import List , Dict, Any
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, EmailStr

class Channel(BaseModel):
    phone: str
    email: Optional[EmailStr]

class Contact(BaseModel):
    name: str
    document: str
    project: str
    protocol: str
    channel: Channel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_ai_message: AIMessage
    last_human_message: HumanMessage


class ToolCallSchema(BaseModel):
    tool_call_id: str
    content: str

class ToolCallResponseSchema(BaseModel):
    data: List[ToolCallSchema]
    contact: Contact

class ToolCallRequestSchema(BaseModel):
    tool_calls: List[ToolCallSchema]
    webhook_url: str
    contact: Contact

class MessageResponseSchema(BaseModel):
    data: str
    contact: Contact

class MessageRequestSchema(BaseModel):
    message: str
    webhook_url: str
    contact: Contact

class UsuarioSchema(BaseModel):
    id: int
    nome: str
    document: str
    phone: str
    email: EmailStr
    project: str
    protocol: str
    processing: bool
    messages: List[Dict[str, Any]]
    class Config:
        orm_mode=True
