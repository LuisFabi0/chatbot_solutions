import os
import operator
import time
from typing import Annotated, TypedDict, List
import re
import uuid
import requests

# --- Dependências Essenciais ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- Dependências da CrewAI ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from bs4 import BeautifulSoup

# ETAPA 1: CONSTANTES, CONHECIMENTO BASE E PROMPT PRINCIPAL

ROBBU_KNOWLEDGE_BASE = """
Nossa história: Fundada em 2016, a Robbu é líder em soluções de automação de comunicação. Nosso produto permite que clientes organizem o relacionamento com seus consumidores em uma solução omnichannel, totalmente personalizável e integrada a diversos sistemas.
Sinergia e tecnologia: Acreditamos que o sucesso está na comunicação personalizada. Combinamos nosso DNA de customer experience com o melhor da tecnologia para transformar o contato entre marcas e clientes.
Parcerias e Alcance: Somos parceiros do Google, Meta e Microsoft, provedores oficiais do WhatsApp Business API e fomos Empresa Destaque do Facebook (Meta) em 2021. Temos mais de 800 clientes em 26 países.
Principais produtos: Plataforma omnichannel (WhatsApp, Instagram, etc.), Chatbots com IA, Automação de marketing, Relatórios, Integrações com CRMs/ERPs, e uma API robusta.
Segurança: Seguimos rigorosamente as normas da LGPD.

Produtos Robbu:
Invenio Live: Plataforma que unifica o atendimento de 10 canais diferentes em um só lugar.
Invenio Center: Ferramenta para campanhas de marketing ativo e análise de métricas.
IDR Chatbot Studio: Estúdio para criar chatbots e automatizar o atendimento.
Positus Messenger: Permite que vários computadores usem o mesmo número de WhatsApp para atendimento.
Carteiro Digital: Envia boletos e documentos de forma automática por WhatsApp ou API.
WebChat: Adiciona um chat em tempo real para conversar com os visitantes do seu site.
Vídeo Chamada: Realiza atendimentos e conferências por vídeo com gravação.
Insights: Ferramenta de análise de dados para acompanhar os resultados do seu atendimento.

***Sempre que o usuário perguntar sobre qualquer um tópico técnico (escopo) abaixo, execute a ferramenta 'pesquisa_tecnica_avancada_robbu' para buscar a documentação relevante que irá fornecer o contexto necessário para a resposta.***
<Assuntos Técnicos que abordamos:>
Criação Templates WhatsApp: Criar templates de WhatsApp de forma rápida e fácil através do nosso novo construtor.
Edição de Tags (Live): Utilizar e gerenciar tags (hashtags) no Invenio Live e Center para identificar e localizar contatos rapidamente.
Configurações Gerais da Conta: Gerenciar as configurações gerais da conta, incluindo informações de perfil e preferências de notificação.
Gerenciar Host de Acesso: Controlar o acesso dos usuários aos diferentes hosts da plataforma.
Como Criar Agendamento (Live): Passo a passo para criar agendamentos no Invenio Live.
Carteiro Digital API: Documentação da API do Carteiro Digital para integração com outros sistemas.
Carteiro Digital: Funcionalidades e benefícios do Carteiro Digital.
Gestão de Frases Prontas: Criar e gerenciar frases prontas para uso no atendimento.
Restrições: Definir restrições.
Campanhas: Público e Importação: Criar e gerenciar campanhas de marketing, incluindo segmentação de público e importação de contatos.
Criando Campanha SMS: Passo a passo para criar campanhas de SMS no Invenio.
Bibliotecas de Mídias: Gerenciar bibliotecas de mídias para uso em campanhas.
Criar Campanhas de WhatsApp: Passo a passo para criar campanhas de WhatsApp no Invenio.
Canais de Atendimento/Canais: Gerenciar canais de atendimento e suas configurações.
Canal WhatsApp: Configurar e gerenciar o canal do WhatsApp no Invenio.
Como Alterar Imagem da Linha WhatsApp: Passo a passo para alterar a imagem da linha do WhatsApp.
Criação de Contatos Invenio Center: Criar e gerenciar contatos na plataforma Invenio Center.
Exportação de Conversas (Live): Exportar conversas do Invenio Live para análise e relatórios.
Fila de Atendimento: Gerenciar a fila de atendimento e priorizar atendimentos.
Filtros de Busca de Contatos: Utilizar filtros avançados para buscar contatos de forma eficiente.
Lista de Desejos (Live): Criar e gerenciar uma lista de desejos no Invenio Live.
Métodos de Distribuição (Live), preditiva e manual: Configurar métodos de distribuição de mensagens no Invenio Live.
Sessão de 24 Horas no WhatsApp: Entender a sessão de 24 horas no WhatsApp e suas implicações.
Usuários: Gerenciar usuários e permissões na plataforma.
Relatórios: Criar e visualizar relatórios sobre o desempenho e uso da plataforma.
Webchat: Configurar e personalizar webchat.
KPI Eventos: Monitorar e analisar os principais indicadores de desempenho (KPIs) dos eventos.
Privacidade e LGPD: Garantir conformidade com a legislação de proteção de dados (LGPD) e políticas de privacidade.
(Meta) Códigos de Erro da API: Consultar e entender os códigos de erro retornados pela API.
(Meta) Migração de On-Premises para a Nuvem: Planejar e executar a migração de sistemas locais (on-premises) para a nuvem.
(Meta) Contas Comerciais do WhatsApp: Criar e gerenciar contas comerciais do WhatsApp.
(Meta) Números de Telefone (Cloud API): Gerenciar números de telefone na Cloud API do WhatsApp.
(Meta) Configuração de Webhooks: Configurar webhooks para receber eventos do WhatsApp.
(Meta) Documentação da API do WhatsApp: Consultar a documentação oficial da API do WhatsApp.
(Meta) Política de Privacidade do WhatsApp: Entender a política de privacidade do WhatsApp.
(Meta) Política de Uso da API do WhatsApp: Consultar a política de uso da API do WhatsApp.
- Qualquer assunto (tópico técnico) fora da lista acima não é abordado diretamente nesse canal e é necessário falar com um atendente humano.
</Assuntos Técnicos que abordamos>

***Se o usuário perguntar sobre as lideranças da Robbu, responda:***
CEO: Álvaro Garcia Neto
CO-Founder: Helber Campregher
"""

ROBBU_DOCS_CONTEXT = [
    {"name": "Criar Templates WhatsApp", "url": "https://docs.robbu.global/docs/center/como-criar-templates-whatsapp"},
    {"name": "Edição de Tags (Live)", "url": "https://docs.robbu.global/docs/live/edicao-de-tags"},
    {"name": "Configurações Gerais da Conta", "url": "https://robbu.mintlify.app/docs/center/configuracoes-gerais-da%20conta"},
    {"name": "Gerenciar Host de Acesso", "url": "https://docs.robbu.global/docs/center/gerenciar-host-de-acesso#o-que-e-o-gerenciar-hosts-de-acesso"},
    {"name": "Como Criar Agendamento (Live)", "url": "https://docs.robbu.global/docs/live/como-criar-agendamento"},
    {"name": "Carteiro Digital API", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital-api"},
    {"name": "Carteiro Digital", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital"},
    {"name": "Gestão de Frases Prontas", "url": "https://docs.robbu.global/docs/center/gestao-frases-prontas"},
    {"name": "Restrições", "url": "https://docs.robbu.global/docs/center/restricoes"},
    {"name": "Campanhas: Público e Importação", "url": "https://docs.robbu.global/docs/center/campanhas-publico-importacao"},
    {"name": "Criando Campanha SMS", "url": "https://docs.robbu.global/docs/center/criando-campanha-sms"},
    {"name": "Bibliotecas de Mídias", "url": "https://docs.robbu.global/docs/center/bibliotecas-de-midias"},
    {"name": "Criar Campanhas de WhatsApp", "url": "https://docs.robbu.global/docs/center/campanhas-de-whatsapp"},
    {"name": "Canais de Atendimento", "url": "https://docs.robbu.global/docs/center/canais-atendimento"},
    {"name": "Canal WhatsApp", "url": "https://docs.robbu.global/docs/center/canal-whatsapp"},
    {"name": "Como Alterar Imagem da Linha WhatsApp", "url": "https://robbu.mintlify.app/docs/center/como-alterar-imagem-da-linha-whatsapp"},
    {"name": "Criação de Contatos Invenio Center", "url": "https://robbu.mintlify.app/docs/center/criacao-de-contatos-invenio-center"},
    {"name": "Exportação de Conversas (Live)", "url": "https://robbu.mintlify.app/docs/live/exportacao-de-conversas"},
    {"name": "Fila de Atendimento", "url": "https://robbu.mintlify.app/docs/center/fila-de-atendimento"},
    {"name": "Filtros de Busca de Contatos", "url": "https://robbu.mintlify.app/docs/center/filtros-de-busca-de-contatos"},
    {"name": "Lista de Desejos (Live)", "url": "https://robbu.mintlify.app/docs/live/lista-de-desejos"},
    {"name": "Métodos de Distribuição (Live), preditiva e manual", "url": "https://docs.robbu.global/docs/live/metodos-de-distribuicao"},
    {"name": "Sessão de 24 Horas no WhatsApp", "url": "https://robbu.mintlify.app/docs/live/sessao-de-24horas-no-whatsapp"},
    {"name": "Usuários", "url": "https://docs.robbu.global/docs/center/usuarios"},
    {"name": "Rotina de Expurgo", "url": "https://docs.robbu.global/docs/center/usuarios#rotina-de-expurgo-de-usu%C3%A1rios"},
    {"name": "Relatórios", "url": "https://docs.robbu.global/docs/center/relatorios"},
    {"name": "Webchat", "url": "https://docs.robbu.global/docs/center/web-chat"},
    {"name": "KPI Eventos", "url": "https://docs.robbu.global/docs/center/dashboard-kpi-eventos"},
    {"name": "Privacidade e LGPD", "url": "https://docs.robbu.global/docs/center/privacidade-e-protecao"}
]

META_DOCS_CONTEXT = [
    {"name": "Códigos de Erro da API", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/support/error-codes/?locale=pt_BR"},
    {"name": "Migração de On-Premises para a Nuvem", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/guides/migrating-from-onprem-to-cloud?locale=pt_BR"},
    {"name": "Contas Comerciais do WhatsApp", "url": "https://developers.facebook.com/docs/whatsapp/overview/business-accounts"},
    {"name": "Números de Telefone (Cloud API)", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/phone-numbers"},
    {"name": "Configuração de Webhooks", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks"},
    {"name": "Documentação da API do WhatsApp", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/"},
    {"name": "Política de Privacidade do WhatsApp", "url": "https://www.whatsapp.com/legal/privacy-policy?lang=pt_BR"},
    {"name": "Política de Uso da API do WhatsApp", "url": "https://www.whatsapp.com/legal/business-policy?lang=pt_BR"}
    ]

# Prompt principal do agente
PROMPT_TEXT = """
Você é o agente help desk especialista da Robbu. Você é um agente profissional, treinado para responder perguntas técnicas sobre a plataforma Robbu e a API do WhatsApp da Meta.

<Apresentação>
- Na primeira interação, você se apresenta como o agente help desk da Robbu e oferece auxílio para a resolução de problemas e o esclarecimento de dúvidas sobre os produtos Robbu.
- Se o usuário iniciar a interação fazendo uma pergunta específica, você deve responder de forma clara e objetiva, utilizando exemplos práticos sempre que possível.
</Apresentação>

<Persona e Tom de Voz>
- Você é profissional, prestativo e direto.
- Comunique-se de forma clara e objetiva. Use a primeira pessoa do plural ('nossos') ao falar sobre a Robbu e trata o interlocutor como 'você'.
- Use gírias moderadas (exemplo: "legal", "bacana", "maravilha", "super") e emojis simples (✅✨😊😉📎📑✉️).
- Evitar jargões técnicos desnecessários e linguagem excessivamente rebuscada. Usar uma comunicação clara, amigável e acessível, como um profissional atencioso que fala a linguagem do cliente.
- Deve ser objetivo, mas fornecer explicações suficientes quando as dúvidas forem mais complexas. Respostas curtas, mas informativas.
</Persona e Tom de Voz>

<Perfil dos Clientes>
- O usuário é um cliente da Robbu que busca ajuda técnica ou informações sobre a plataforma.
- Usuários que precisam esclarecer dúvidas sobre a plataforma Robbu, funcionalidades como 'templates' e 'relatórios', ou sobre a API do WhatsApp, como 'códigos de erro' e 'contas comerciais'.
- Os usuários podem ter diferentes níveis de conhecimento técnico, desde iniciantes até desenvolvedores experientes.
- Os usuários podem estar buscando soluções rápidas para problemas comuns ou informações detalhadas sobre funcionalidades específicas.
- Os usuários podem estar frustrados, nesse caso, é importante ouvir suas reclamações e fornecer suporte adequado.
</Perfil dos Clientes>

<Restrições Gerais>
- Você não responde perguntas pessoais, políticas ou que não estejam relacionadas à Robbu, não responde sobre outros produtos ou serviços, e não especula sobre assuntos desconhecidos.
- Se o assunto estiver completamente fora do escopo da Robbu, informe educadamente que não pode ajudar com aquele tópico.
- Você só deve falar sobre a Robbu e suas funcionalidades, não deve falar sobre produtos ou serviços de terceiros como blip, mundiale e etc.
- Não realiza calculos financeiros, nem calculos basicos como 2+2, nem perguntas de lógica ou enigmas.
- Não responde sobre questões polemicas, como política ou religião.
- Não falar sobre assuntos relacionados a medicina ou condições de saude.
- Não deve falar sobre a CrewAI, nem sobre o que é um agente, nem sobre como funciona a CrewAI.
- Não deve falar sobre o que é um LLM, nem sobre como funciona o modelo de linguagem.
- Não ofereça dicas de economia ou finanças, nesse caso, informe que é necessário (1)falar com um atendente humano ou (2)entrar em contato com o comercial comercial@robbu.com.br.
- Não deve falar sobre o que é um assistente virtual, nem sobre como funciona um assistente virtual.
- Caso o usuário faça perguntas que não se encaixam dentro do escopo dos tópicos técnicos, informe que é necessáriofalar com um atendente humano para sanar essa dúvida.
- Não solicite prints, nem imagens.
- A base de conhecimento da Robbu é a sua principal referência sobre os assuntos técnicos abordados nesse canal, consulte para validar se aborda o tema que responderia a dúvida do usuário. Qualquer tema que não estiver sendo citado na base de conhecimento diga que não possui a informação e pergunte se o usuário gostaria de falar com um atendente humano.
Exemplo: "O que é X?", "Como configurar Y?", "Quais são os benefícios de W?", "Me ajuda com Z?"> Consulte a base de conhecimento > Se não encontrar o tópico/tema X, informe que não possui a informação e pergunte se o usuário gostaria de falar com um atendente humano.
- Quando abordar um assunto fora do escopo da Robbu, informe educadamente que não pode ajudar com aquele tópico e oriente o usuário a buscar informações em com o canal correto ou setor apropriado.
- Quando não conseguir obter informações para responder o usuário, pergunte se ele gostaria de falar com um atendente humano, se sim USE a ferramenta 'falar_com_atendente_humano'.
- Caso o usuário deseje orientações para abrir um chamado ou deseje falar diretamente com um atendente, utilize a ferramenta 'falar_com_atendente_humano'.
- Se o usuário quiser saber como abrir um chamado, USE a ferramenta 'falar_com_atendente_humano' para que um atendente humano possa auxiliá-lo.
- Se o usuário insistir em obter informações que você não pode fornecer, mantenha a postura profissional e reforce que não pode ajudar com aquele tópico e pergunte se ele gostaria de ser direcionado para um atendente humano.
- Se o usuário demonstrar estar muito frustrado ou irritado com uma situação (por exemplo: insultos, xingamentos, ameaças, ou apenas dizendo que a solicitação dele não está sendo atendida de nenhuma forma) USE a ferramenta 'falar_com_atendente_humano' para que um atendente humano possa auxiliá-lo.
- Não faça susgestões de melhoria ou peça feedback sobre a Robbu.
- Não sugira coisas que não estejam no seu conhecimento base. Exemplo: Não sugira fornecer exemplos praticos ou passo a passo se o conhecimento base não abordar o assunto.
- Nunca peça para o usuário aguardar ou pedir para esperar.
</Restrições Gerais>


**Seu Conhecimento Base sobre a Robbu (Responda diretamente se a resposta da sua pergunta estiver aqui):**
{robbu_knowledge}

**Suas Ferramentas:**
Você tem acesso a ferramentas para buscar informações que não estão no seu conhecimento base.
Você **DEVE** decidir, com base na pergunta do usuário, se pode responder diretamente ou se precisa usar uma ferramenta.
- Para perguntas técnicas sobre funcionalidades específicas (templates, relatórios, integrações, canais de atendimento, Configurações, Métodos de Distribuição preditiva e manual e etc...), APIs, ou erros, **USE** a ferramenta `pesquisa_tecnica_avancada_robbu`.
- Para saudações, agradecimentos, ou perguntas institucionais simples (cobertas no seu conhecimento base), **NÃO USE** ferramentas, responda diretamente usando seu conhecimento base.
- Se o usuário optar por ser transferido para um atendente humano, **USE** a ferramenta `falar_com_atendente_humano`.

**Fluxo da Conversa:**

1. Analise a pergunta do usuário.
2. Se a resposta estiver no seu "Conhecimento Base", responda diretamente.
3. Se for uma pergunta técnica complexa sobre a plataforma Robbu, configurações ou a API do WhatsApp, chame a ferramenta `pesquisa_tecnica_avancada_robbu` e use o resultado para formular sua resposta final.
4. A resposta final deve ser explicativa e clara, abordando os pontos principais para que o usuário entenda a solução ou informação fornecida. Se possivel no formato de passo a passo. Caso seja necessário que a resposta contenha exemplos em codigos não formatados, formate sem identar de forma explicativa. Se o retorno da ferramenta for N/A ou um erro, informe que não foi possível localizar a informação e pergunte se o usuário gostaria de falar com um atendente humano.
""".format(robbu_knowledge=ROBBU_KNOWLEDGE_BASE)


# ETAPA 2: CONFIGURAÇÕES


def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Chave da API da OpenAI não encontrada. Verifique seu arquivo .env")
    return api_key

API_KEY = load_api_key()
LLM = ChatOpenAI(model="gpt-4.1", api_key=API_KEY, temperature=0.7)
VALIDATOR_LLM = ChatOpenAI(model="gpt-4.1-mini", api_key=API_KEY, temperature=0.0)
OPENAI_CLIENT = OpenAI(api_key=API_KEY)


# ETAPA 3: FERRAMENTAS
class EnhancedWebScrapeTool(BaseTool):
    name: str = "Extração Avançada de Conteúdo Web"
    description: str = "Extrai conteúdo de páginas web com tratamento de erros e formatação."
    def _run(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]): element.decompose()
            main_content = (soup.find('main') or soup.find('article') or soup.find('body'))
            text = main_content.get_text("\n", strip=True) if main_content else ""
            return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])[:5500]
        except Exception as e:
            return f"[ERRO_EXTRACAO:{str(e)}]"

class TechnicalCrewExecutor:
    def run(self, query: str) -> str:
        analisador = Agent(role="Analisador de Documentos", goal="Analisar a pergunta e encontrar a URL mais relevante.", backstory="Especialista em encontrar o documento certo.", llm=LLM, verbose=False)
        extrator = Agent(role="Extrator de Conteúdo", goal="Extrair o conteúdo de uma página web.", backstory="Especialista em parsing de HTML.", llm=LLM, tools=[EnhancedWebScrapeTool()], verbose=False)
        redator = Agent(role="Redator Técnico", goal="Produzir respostas claras, objetivas e profissionais.", backstory="Especialista em suporte técnico.", llm=LLM, verbose=False)
        
        documentos_formatados = "\n".join([f"- Título: '{doc['name']}', URL: {doc['url']}" for doc in ROBBU_DOCS_CONTEXT + META_DOCS_CONTEXT])

        tarefa_analise = Task(description=f"Analise a pergunta: '{query}'. Encontre a URL mais relevante na lista:\n{documentos_formatados}", expected_output="A URL exata ou 'N/A'.", agent=analisador)
        crew_analise = Crew(agents=[analisador], tasks=[tarefa_analise], process=Process.sequential)
        url = crew_analise.kickoff()
        if "N/A" in url or not str(url).startswith("http"): return "Não localizei uma página específica para essa dúvida em nossa documentação."

        tarefa_extracao = Task(description=f"Extraia o conteúdo da URL: {url}", expected_output="O texto limpo da página.", agent=extrator)
        tarefa_redacao = Task(description=f"Produza uma resposta para '{query}' usando o conteúdo extraído.", expected_output="Resposta técnica.", agent=redator, context=[tarefa_extracao])
        
        crew_processamento = Crew(agents=[extrator, redator], tasks=[tarefa_extracao, tarefa_redacao], process=Process.sequential)
        return crew_processamento.kickoff() or f"Não foi possível processar a solicitação para a URL: {url}"

@tool
def pesquisa_tecnica_avancada_robbu(query: str) -> str:
    """Use para responder a perguntas técnicas sobre a plataforma Robbu, funcionalidades, ou a API do WhatsApp."""
    return TechnicalCrewExecutor().run(query)

@tool
async def falar_com_atendente_humano() -> str:
    """Sinaliza a necessidade de transferir o atendimento para um atendente humano."""
    return "Transferência para atendimento humano solicitada."

@tool
async def finalizar_conversa(motivo: str) -> str:
    """Use para encerrar a conversa quando o usuário insistir em um assunto fora do escopo."""
    return f"Conversa finalizada pelo motivo: {motivo}"

# ETAPA 4: GRAFO DE ORQUESTRAÇÃO

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- Guardrails de Entrada ---

# --- Guardrail de Moderação de Conteúdo ---
async def content_moderation_guardrail(user_query: str) -> bool:
    """
    Verifica se a entrada do usuário contém conteúdo inadequado usando a API de Moderação da OpenAI.
    Retorna True se o conteúdo for inadequado, False caso contrário.
    """
    try:
        response = await OPENAI_CLIENT.moderations.create(input=user_query)
        is_flagged = response.results[0].flagged
        print(f"--- GUARDRAIL DE MODERAÇÃO: Veredito: {'INADEQUADO' if is_flagged else 'OK'} para '{user_query}' ---")
        return is_flagged
    except Exception as e:
        print(f"--- ERRO NO GUARDRAIL DE MODERAÇÃO: {e} ---")
        return False ### Fail-safe: Em caso de erro, não bloqueia o usuário. ###

# --- Guardrail de Detecção de PII ---
async def pii_detection_guardrail(user_query: str) -> (bool, str):
    """
    Detecta e mascara informações PII usando Regex.
    Retorna uma tupla: (True se PII foi encontrado, texto com PII mascarado).
    """
    pii_patterns = {
        'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'CPF': r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
        'CARTAO_CREDITO': r'\b(?:\d[ -]*?){13,16}\b',
        'TELEFONE': r'\b(?:\+?55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b'
    }
    
    sanitized_query = user_query
    pii_found = False
    
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, sanitized_query):
            pii_found = True
            sanitized_query = re.sub(pattern, f"[{pii_type}_REMOVIDO]", sanitized_query)

    if pii_found:
        print(f"--- GUARDRAIL DE PII: PII detectado. Query original: '{user_query}'. Query sanitizada: '{sanitized_query}' ---")
    
    return pii_found, sanitized_query

# Guardrail de Tópico 
async def input_topic_guardrail(user_query: str) -> bool:

    class TopicValidation(BaseModel):
        decision: str = Field(description="Decida se o tópico é 'DENTRO DO ESCOPO' ou 'FORA DO ESCOPO'.")

    validator_llm = VALIDATOR_LLM.with_structured_output(TopicValidation)
    
    prompt = f"""
    Sua tarefa é classificar se a pergunta do usuário está relacionada aos produtos,
    serviços ou documentação técnica da Robbu. Ignore saudações ou despedidas.
    Se for uma saudação (Oi, Olá e etc.) ou agradecimento (Obrigado, Me ajudou muito, Salvou e etc.), classifique como 'DENTRO DO ESCOPO'.
    Se for uma pergunta institucional simples Ex: "O que é Robbu" (coberta no conhecimento base), classifique como 'DENTRO DO ESCOPO'.
    Se a pergunta for ambígua (ex: "pode me ajudar?", "não sei o que fazer", "não tenho certeza", "Como posso fazer isso"), classifique como 'DENTRO DO ESCOPO' para permitir que o agente principal peça mais esclarecimentos.
    Se for pergunta tecnica, precisa ser extritamente baseada na base de conhecimento da Robbu, se o assunto não estiver dentro dos tópicos abordados na base de conhecimento, a resposta deve ser FORA DO ESCOPO.
    
    Base de Conhecimento para referência de escopo:
    ---
    {ROBBU_KNOWLEDGE_BASE}
    ---
    
    Pergunta do Usuário: 
    ---
    "{user_query}"
    ---

    O tópico está 'DENTRO DO ESCOPO' ou 'FORA DO ESCOPO'?
    """
    try:
        result = validator_llm.invoke(prompt)
        print(f"--- GUARDRAIL DE TÓPICO (Tagging): Veredito: {result.decision} para '{user_query}' ---")
        return result.decision == "FORA DO ESCOPO"
    except Exception as e:
        print(f"--- ERRO NO GUARDRAIL DE TÓPICO: {e} ---")
        return False

# --- Guardrail de Saída ---
async def factual_guardrail(message_to_validate: AIMessage, user_query: str) -> AIMessage:
    if not message_to_validate.content:
        return message_to_validate

    class ValidationResult(BaseModel):
        decision: str = Field(description="A decisão, deve ser 'APROVADO' ou 'REPROVADO'.")
        reason: str = Field(description="Uma breve explicação para a decisão.")

    validator_llm_with_tool = VALIDATOR_LLM.with_structured_output(ValidationResult)
    
    validator_prompt = f"""
    Você é um verificador de qualidade rigoroso para um assistente de help desk da empresa Robbu.
    Sua única tarefa é verificar se a resposta fornecida é factual, consistente e estritamente alinhada com os assuntos disponiveis na base de conhecimento da Robbu, considerando a pergunta original do usuário.

    A resposta NÃO deve conter:
    - Especulações ou informações não confirmadas.
    - Opiniões pessoais.
    - Conteúdo fora do escopo dos produtos e serviços da Robbu.
    - Promessas ou garantias que não podem ser cumpridas.
    - Exemplo: "Sim, você pode fazer X" se X não for suportado pela Robbu.
    - Se após o assistente não encontrar a informação e não informar que não foi possível localizar a informação e perguntar se o usuário gostaria de falar com um atendente humano.
    - Não deve sugerir melhorias ou pedir feedback sobre a Robbu.
    - Não deve sugerir coisas que não estejam no seu conhecimento base.
    - Se for pergunta tecnica, a resposta precisa ser extritamente baseada na base de conhecimento da Robbu, se o assunto não estiver dentro dos tópicos abordados na base de conhecimento, a resposta deve ser REPROVADO.
    - Algumas perguntas podem ser relacionadas a Meta (Facebook) e a API do WhatsApp, nesse caso, a resposta precisa ser extritamente baseada na base de conhecimento da Robbu, se o assunto não estiver dentro dos tópicos abordados na base de conhecimento, a resposta deve ser REPROVADO.
    
    Pergunta do usuário:
    ---
    {user_query}
    ---

    Resposta do assistente para validar:
    ---
    {message_to_validate.content}
    ---

    Base de conhecimento da Robbu para referência:
    ---
    {ROBBU_KNOWLEDGE_BASE}
    ---
    
    Baseado na sua análise, decida se a resposta é 'APROVADO' ou 'REPROVADO'.
    """
    
    try:
        judge_result = validator_llm_with_tool.invoke(validator_prompt)
        print(f"--- GUARDRAIL DE SAÍDA: Veredito: {judge_result.decision}, Razão: {judge_result.reason} ---")

        if judge_result.decision == "REPROVADO":
            print("--- GUARDRAIL AÇÃO: Fallback acionado. Gerando chamada para 'falar_com_atendente_humano'. ---")
            tool_call_id = f"tool_{uuid.uuid4()}"
            fallback_tool_call = {
                "name": "falar_com_atendente_humano",
                "args": {},
                "id": tool_call_id
            }
            return AIMessage(content="", tool_calls=[fallback_tool_call])
            
    except Exception as e:
        print(f"--- ERRO NO GUARDAIL DE SAÍDA: {e}. Acionando fallback por segurança. ---")
        tool_call_id = f"tool_{uuid.uuid4()}"
        fallback_tool_call = {
            "name": "falar_com_atendente_humano",
            "args": {},
            "id": tool_call_id
        }
        return AIMessage(content="", tool_calls=[fallback_tool_call])

    return message_to_validate

# --- Nó do Grafo Modificado ---
async def agent_node(state: AgentState) -> dict:
    """
    Nó principal que aplica guardrails de segurança e relevância antes de processar.
    """
    print("\n--- NÓ: Agente ---")
    messages = state['messages']
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        user_query = last_message.content
        
        # 1. Guardrail de Moderação de Conteúdo 
        if await content_moderation_guardrail(user_query):
            response_text = "Não posso processar esta solicitação. Se precisar de ajuda com os produtos Robbu, por favor, reformule sua pergunta sem usar linguagem inadequada."
            return {"messages": [AIMessage(content=response_text)]}

        # 2. Guardrail de Detecção de PII 
        pii_found, sanitized_query = await pii_detection_guardrail(user_query)
        if pii_found:
            # Atualiza a mensagem no estado para a versão sanitizada ANTES de retornar
            last_message.content = sanitized_query
            last_message.additional_kwargs['pii_detected_note'] = "Nota: Dados sensíveis foram removidos da pergunta do usuário por segurança."

            print("--- GUARDRAIL DE PII: PII detectado. Acionando transferência para atendente humano. ---")
            tool_call_id = f"tool_{uuid.uuid4()}"
            fallback_tool_call = {
                "name": "falar_com_atendente_humano",
                "args": {},
                "id": tool_call_id
            }
            # Retorna a chamada da ferramenta sem mensagem adicional.
            return {"messages": [AIMessage(content="", tool_calls=[fallback_tool_call])]}

        # 3. Guardrail de Tópico (só executa se os anteriores passarem)
        if 'topic' not in last_message.additional_kwargs:
            if await input_topic_guardrail(user_query):
                last_message.additional_kwargs['topic'] = 'off_topic'
            else:
                last_message.additional_kwargs['topic'] = 'on_topic'


    consecutive_off_topic = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if msg.additional_kwargs.get('topic') == 'off_topic':
                consecutive_off_topic += 1
            else:
                break
    
    print(f"--- NÓ AGENTE: Strikes consecutivos (via tags): {consecutive_off_topic} ---")

    if consecutive_off_topic >= 3:
        print("--- NÓ AGENTE: Limite de insistência atingido. Acionando finalização. ---")
        tool_call = {
            "name": "finalizar_conversa",
            "args": {"motivo": "Usuário insistiu em assunto fora do escopo."},
            "id": f"tool_{uuid.uuid4()}"
        }
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

    # Continua o fluxo normal
    print("--- NÓ AGENTE: Gerando resposta... ---")
    response = CHAIN.invoke({"messages": messages})
    
    if response.tool_calls:
        print("--- NÓ AGENTE: Modelo decidiu usar uma ferramenta. Pulando guardrail de saída. ---")
        return {"messages": [response]}
    else:
        last_human_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_query = msg.content
                break
        
        print("--- NÓ AGENTE: Enviando resposta para o guardrail de saída. ---")
        validated_response = await factual_guardrail(response, last_human_query)
        return {"messages": [validated_response]}

def route_action(state: AgentState) -> str:
    last_message = state["messages"][-1]
    print(f"--- ROTEADOR: Analisando a mensagem do tipo {type(last_message).__name__} ---")

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("--- ROTEADOR: Sem chamada de ferramenta. Terminando o fluxo. ---")
        return END

    for tool_call in last_message.tool_calls:
        if tool_call["name"] in ["falar_com_atendente_humano", "finalizar_conversa"]:
            print(f"--- ROTEADOR: Ferramenta de término ({tool_call['name']}) detectada. Finalizando. ---")
            return END
    
    print("--- ROTEADOR: Chamada de ferramenta detectada. Direcionando para a ação. ---")
    return "action"


def build_graph():
    """Constrói e compila o grafo."""
    workflow = StateGraph(AgentState)
    
    TOOLS = [pesquisa_tecnica_avancada_robbu, falar_com_atendente_humano, finalizar_conversa]
    TOOL_EXECUTOR = ToolNode(TOOLS)
    MODEL = LLM.bind_tools(TOOLS)
    
    global PROMPT, CHAIN
    PROMPT = ChatPromptTemplate.from_messages([("system", PROMPT_TEXT), MessagesPlaceholder(variable_name="messages")])
    CHAIN = PROMPT | MODEL

    workflow.add_node("agent", agent_node)
    workflow.add_node("action", TOOL_EXECUTOR)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        route_action, 
        {
            "action": "action", 
            END: END          
        },
    )
    workflow.add_edge("action", "agent")
    
    return workflow.compile()

APP = build_graph()
