import os
import operator
import ast 
from typing import Annotated, TypedDict, List

# --- Dependências Essenciais ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import requests
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
from requests.exceptions import HTTPError


# --- Dependências da CrewAI ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# ETAPA 1: DEFINIÇÕES GLOBAIS E AGENTES

# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Chave da API da OpenAI não encontrada. Verifique seu arquivo .env")

# Base de conhecimento sobre a Robbu para perguntas gerais
ROBBU_KNOWLEDGE_BASE = """
Nossa história: Fundada em 2016, a Robbu é líder em soluções de automação de comunicação entre marcas e seus clientes. Nosso produto permite que nossos clientes organizem o relacionamento com seus clientes em uma solução omnichannel, totalmente personalizável, que se integra aos serviços, sistemas e APIs de cada um dos nossos clientes e parceiros. O melhor de tudo? Uma solução com interface intuitiva, capaz de aumentar as vendas, crescer a base de clientes e reduzir os custos operacionais e que conecta, no mesmo lugar, inteligência artificial, chatbots e atendimento humano.
Sinergia entre tecnologia e conexões: Nascemos com a crença de que o sucesso de uma empresa está na comunicação e no relacionamento personalizado que ela possui com seus clientes. Ao combinar nosso DNA de customer experience com o que há de melhor em atendimento e vendas no mundo digital, criamos a Robbu, uma solução inteligente que transforma o contato de marcas com seus clientes.
Parcerias e Alcance: A empresa é parceira do Google, Meta e Microsoft, provedora oficial do WhatsApp Business API e foi a Empresa Destaque do Facebook (Meta) em 2021. Com sede em São Paulo, Brasil, e presente em outros três países: Argentina, Portugal e Estados Unidos, a Robbu conta com mais de 800 clientes e parceiros de negócios em 26 países.

***Apenas se perguntarem sobre a segurança:*** A Robbu segue corretamente as normas da Lei Geral de Proteção de Dados***
"""

# Define o LLM que será usado em todo o script
llm = ChatOpenAI(model="gpt-4.1", api_key=API_KEY)

# --- Definição das Ferramentas como Classes ---
class RobbuDocsSearchTool(BaseTool):
    name: str = "Pesquisa na Documentação Robbu"
    description: str = "Busca na documentação e no site oficial da Robbu para encontrar artigos, tutoriais e documentação de API."
    def _run(self, query: str) -> str:
        site_specific_query = f"site:robbu.global OR site:docs.robbu.global {query}"
        print(f"--- [CREW TOOL] Executando busca focada: {site_specific_query} ---")
        return DuckDuckGoSearchAPIWrapper().run(site_specific_query)

class ScrapeWebsiteTool(BaseTool):
    name: str = "Extração de Conteúdo de Website"
    description: str = "Acessa uma URL e extrai o conteúdo de texto principal da página."
    def _run(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            text = main_content.get_text(separator="\n", strip=True) if main_content else soup.get_text(separator="\n", strip=True)
            return text[:4000]
        except Exception as e:
            return f"Erro ao processar a URL {url}: {e}"

# --- Contexto de URLs da Documentação ---
DOCUMENT_URLS_CONTEXT = [
    {"name": "Templates WhatsApp", "url": "https://docs.robbu.global/docs/center/como-criar-templates-whatsapp"},
    {"name": "Edição de Tags (Live )", "url": "https://docs.robbu.global/docs/live/edicao-de-tags"},
    {"name": "Configurações Gerais da Conta", "url": "https://robbu.mintlify.app/docs/center/configuracoes-gerais-da%20conta"},
    {"name": "Gerenciar Host de Acesso", "url": "https://docs.robbu.global/docs/center/gerenciar-host-de-acesso#o-que-e-o-gerenciar-hosts-de-acesso"},
    {"name": "Como Criar Agendamento (Live )", "url": "https://docs.robbu.global/docs/live/como-criar-agendamento"},
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
    {"name": "Exportação de Conversas (Live )", "url": "https://robbu.mintlify.app/docs/live/exportacao-de-conversas"},
    {"name": "Fila de Atendimento", "url": "https://robbu.mintlify.app/docs/center/fila-de-atendimento"},
    {"name": "Filtros de Busca de Contatos", "url": "https://robbu.mintlify.app/docs/center/filtros-de-busca-de-contatos"},
    {"name": "Lista de Desejos (Live )", "url": "https://robbu.mintlify.app/docs/live/lista-de-desejos"},
    {"name": "Métodos de Distribuição (Live )", "url": "https://docs.robbu.global/docs/live/metodos-de-distribuicao"},
    {"name": "Sessão de 24 Horas no WhatsApp", "url": "https://robbu.mintlify.app/docs/live/sessao-de-24horas-no-whatsapp"},
    {"name": "Usuários", "url": "https://docs.robbu.global/docs/center/usuarios"},
    {"name": "Relatórios", "url": "https://docs.robbu.global/docs/center/relatorios"},
    {"name": "Webchat", "url": "https://docs.robbu.global/docs/center/web-chat"},
    {"name": "KPI Eventos", "url": "https://docs.robbu.global/docs/center/dashboard-kpi-eventos"},
    {"name": "Privacidade e LGPD", "url": "https://docs.robbu.global/docs/center/privacidade-e-protecao"}
]

# --- Definição dos Agentes ---
analisador = Agent(
    role="Analisador de Requisitos de Usuário",
    goal="Analisar a pergunta de um usuário, identificar todas as questões distintas e mapear cada uma para a URL mais relevante de uma lista de documentos.",
    backstory="Você é especialista em decompor perguntas complexas em partes individuais e encontrar a documentação exata para cada parte.",
    llm=llm,
    verbose=True
)
pesquisador = Agent(
    role="Pesquisador de Documentação Sênior",
    goal="Encontrar a URL mais relevante na documentação da Robbu para responder a uma pergunta específica.",
    backstory="Você é um mestre em usar a busca para encontrar informações precisas.",
    tools=[RobbuDocsSearchTool()],
    llm=llm,
    verbose=True
)
extrator = Agent(
    role="Especialista em Extração de Conteúdo",
    goal="Acessar uma URL e extrair seu texto limpo e útil.",
    backstory="Sua habilidade é ler uma página da web e extrair apenas a informação essencial.",
    tools=[ScrapeWebsiteTool()],
    llm=llm,
    verbose=True
)
redator = Agent(
    role="Redator de Suporte Técnico",
    goal="Criar uma resposta clara e precisa para o usuário, baseada no conteúdo técnico fornecido.",
    backstory="Você transforma jargão técnico em respostas fáceis de entender formatadas de forma personalizada para o usuário final.",
    llm=llm,
    verbose=True
)
print("✅ Agentes da Crew criados e rodando.")

# ETAPA 2: LÓGICA DO GRAFO E DAS CREWS

class OrchestratorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def executar_robbu_crew(question: str) -> str:
    documentos_formatados = "\n".join([str(doc) for doc in DOCUMENT_URLS_CONTEXT])
    
    tarefa_analise = Task(
        description=f"""
        Analise a pergunta do usuário: "{question}".
        Sua tarefa é identificar cada pergunta distinta e encontrar a URL mais relevante para cada uma na lista de 'Documentos Conhecidos'. Siga estritamente estas regras:
        1. **Busca por Correspondência Direta:** Primeiro, tente encontrar um documento cujo 'name' corresponda diretamente aos termos na pergunta.
        2. **Busca por Similaridade Semântica (Plano B):** Se NENHUMA correspondência direta for encontrada, avalie os documentos por similaridade de conceito.
        3. **Falha na Busca:** Se, após as duas tentativas, nenhum documento relevante for encontrado, você DEVE indicar a falha.
        **Restrição Absoluta:** Você SÓ PODE usar a lista de 'Documentos Conhecidos' abaixo.
        **Documentos Conhecidos para sua análise:**
        {documentos_formatados}
        """,
        expected_output="""
        Uma string de lista Python, contendo um dicionário para cada pergunta identificada com as chaves 'pergunta' e 'url'.
        - Se um documento relevante for encontrado, o valor de 'url' DEVE ser a URL real do documento.
        - Se NENHUM documento relevante for encontrado, o valor de 'url' DEVE ser None.
        Exemplo de sucesso: "[{'pergunta': 'Como funciona a janela de 24hrs?', 'url': 'https://robbu.mintlify.app/docs/live/sessao-de-24horas-no-whatsapp'}]"
        Exemplo de falha: "[{'pergunta': 'Como faço para integrar com o sistema SAP?', 'url': None}]"
        """,
        agent=analisador
    )
    
    print("\n--- [ETAPA 1/3] Iniciando análise da pergunta... ---")
    crew_analise = Crew(agents=[analisador], tasks=[tarefa_analise], verbose=False)
    resultado_bruto_obj = crew_analise.kickoff()
    resultado_analise_str = str(resultado_bruto_obj).strip().strip('"')
    try:
        lista_de_tarefas = ast.literal_eval(resultado_analise_str)
        if not isinstance(lista_de_tarefas, list):
            raise ValueError("Análise não retornou uma lista.")
    except (ValueError, SyntaxError) as e:
        print(f"Erro ao processar o resultado da análise: {e}. Usando a pergunta original.")
        tarefa_unica_pesquisa = Task(description=f"Encontre a melhor URL para a pergunta: {question}", agent=pesquisador, expected_output="A URL mais relevante.")
        crew_pesquisa = Crew(agents=[pesquisador], tasks=[tarefa_unica_pesquisa])
        url_unica = str(crew_pesquisa.kickoff())
        lista_de_tarefas = [{'pergunta': question, 'url': url_unica}]
    print(f"\n--- [ETAPA 2/3] Processando {len(lista_de_tarefas)} pergunta(s) identificada(s)... ---")
    respostas_individuais = []
    for i, tarefa in enumerate(lista_de_tarefas):
        sub_pergunta = tarefa.get('pergunta', question)
        url = tarefa.get('url', '')
        if not url or not url.startswith('http'):
            respostas_individuais.append(f"Para sua pergunta '{sub_pergunta}':\nNão consegui encontrar um documento relevante em nossa base de conhecimento para fornecer uma resposta.")
            continue
        print(f"\n--- Processando Sub-Tarefa {i+1}: '{sub_pergunta}' com a URL: {url} ---")
        tarefa_extracao = Task(description=f"Extraia o conteúdo completo da URL: {url}", expected_output="O texto limpo da página.", agent=extrator)
        
        tarefa_redacao = Task(
            description=f"Com base no conteúdo extraído, escreva uma resposta clara e completa para a pergunta: '{sub_pergunta}'",
            expected_output="""
            O texto final da resposta, formatado em português para o usuário.
            A resposta deve ser autossuficiente e diretamente responder à pergunta do usuário, utilizando as informações do contexto.
            NÃO inclua frases como 'Eu agora posso te dar uma ótima resposta' ou qualquer outra confirmação. Apenas forneça a resposta em si.
            """,
            agent=redator,
            context=[tarefa_extracao]
        )
        
        crew_processamento = Crew(agents=[extrator, redator], tasks=[tarefa_extracao, tarefa_redacao], process=Process.sequential, verbose=True)
        resposta_individual = crew_processamento.kickoff()
        respostas_individuais.append(f"Para sua pergunta '{sub_pergunta}':\n{str(resposta_individual)}")
    print("\n--- [ETAPA 3/3] Consolidando respostas... ---")
    if len(respostas_individuais) > 1:
        return "Aqui estão as respostas para as suas perguntas:\n\n" + "\n\n---\n\n".join(respostas_individuais)
    return respostas_individuais[0] if respostas_individuais else "Não foi possível processar sua solicitação."

def classificar_input(state: OrchestratorState):
    last_human_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("type") == "human"):
            last_human_message_content = msg.content if isinstance(msg, HumanMessage) else msg.get("content")
            break
            
    prompt = f"""
    Analise a mensagem do usuário e classifique-a em UMA das seguintes categorias. Seja muito preciso.

    CATEGORIAS:
    1. **saudacao**: Apenas cumprimentos.
    - Exemplos: "oi", "bom dia", "olá"

    2. **agradecimento_confirmacao**: Agradecimentos, despedidas ou confirmações simples.
    - Exemplos: "obrigado", "valeu", "entendi", "ok", "tchau"

    3. **pergunta_simples_institucional**: Perguntas gerais sobre a empresa Robbu, sua história, parcerias, confiança ou o que ela faz.
    - Exemplos: "quem é a robbu?","com quem vocês têm parceria?", "o que a robbu faz?"

    4. **pergunta_tecnica_produto**: Perguntas específicas sobre como usar um produto, funcionalidade, API, ou sobre conceitos técnicos como 'campanhas', 'templates', 'webchat', 'relatórios', 'importação'.
    - Exemplos: "como criar uma campanha?", "me fale sobre a API do carteiro digital", "como configuro o webchat?", "o que são templates de whatsapp?, "Tem Politica de privacidade?", "Me explica X de forma facil".

    5. **fora_de_escopo**: Qualquer outra pergunta que não tenha relação com a empresa Robbu, seus produtos ou serviços.
    - Exemplos: "qual a previsão do tempo?", "quem vai ganhar o jogo?", "me conte uma piada"

    ---
    Mensagem do Usuário para classificar: "{last_human_message_content}"
    ---
    
    A categoria é:
    """
    response = llm.invoke(prompt)
    # Retorna a categoria em minúsculas e sem espaços para facilitar a decisão
    return {"messages": [AIMessage(content=response.content.strip().lower(), name="classification_decision")]}


def saudacao(state: OrchestratorState):
    return {"messages": [AIMessage(content="Help Desk Robbu. Como posso ajudar?")]}

def responder_agradecimento(state: OrchestratorState):
    return {"messages": [AIMessage(content="Se precisar de mais alguma coisa, é só perguntar.")]}

def responder_fora_de_escopo(state: OrchestratorState):
    return {"messages": [AIMessage(content="Desculpe, só posso responder a perguntas sobre a Robbu. Como posso ajudar?")]}

def responder_pergunta_simples(state: OrchestratorState):
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("type") == "human"):
            user_message_content = msg.content if isinstance(msg, HumanMessage) else msg.get("content")
            break
            
    prompt = f"""Use o CONHECIMENTO BASE para responder à pergunta do usuário.
    --- CONHECIMENTO BASE ---
    {ROBBU_KNOWLEDGE_BASE}
    ---
    Pergunta do Usuário: "{user_message_content}" """
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

def chamar_robbu_crew(state: OrchestratorState):
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("type") == "human"):
            question = msg.content if isinstance(msg, HumanMessage) else msg.get("content")
            break
            
    if not question:
        return {"messages": [AIMessage(content="Não consegui identificar sua pergunta.")]}
        
    crew_response = executar_robbu_crew(question)
    return {"messages": [AIMessage(content=crew_response)]}

def decidir_fluxo_inicial(state: OrchestratorState):
    decision = state["messages"][-1].content
    print(f"--- [ROUTER] Decisão de classificação: {decision} ---") # Adicionado para debugging
    if "agradecimento_confirmacao" in decision: return "responder_agradecimento"
    if "fora_de_escopo" in decision: return "responder_fora_de_escopo"
    # Ajustado para os novos nomes de categoria
    if "pergunta_tecnica_produto" in decision: return "chamar_robbu_crew"
    if "pergunta_simples_institucional" in decision: return "responder_pergunta_simples"
    return "saudacao"

# ETAPA 3: CONSTRUÇÃO E COMPILAÇÃO DO GRAFO

workflow = StateGraph(OrchestratorState)

workflow.add_node("classificar_input", classificar_input)
workflow.add_node("saudacao", saudacao)
workflow.add_node("responder_agradecimento", responder_agradecimento)
workflow.add_node("responder_fora_de_escopo", responder_fora_de_escopo)
workflow.add_node("responder_pergunta_simples", responder_pergunta_simples)
workflow.add_node("chamar_robbu_crew", chamar_robbu_crew)

workflow.set_entry_point("classificar_input")
workflow.add_conditional_edges(
    "classificar_input",
    decidir_fluxo_inicial,
    {
        "saudacao": "saudacao", 
        "responder_agradecimento": "responder_agradecimento",
        "responder_fora_de_escopo": "responder_fora_de_escopo",
        "responder_pergunta_simples": "responder_pergunta_simples", 
        "chamar_robbu_crew": "chamar_robbu_crew"
    }
)
workflow.add_edge("saudacao", END)
workflow.add_edge("responder_agradecimento", END)
workflow.add_edge("responder_fora_de_escopo", END)
workflow.add_edge("responder_pergunta_simples", END)
workflow.add_edge("chamar_robbu_crew", END)

# A variável 'app' agora está no escopo global, permitindo a importação
app = workflow.compile()
print("✅ Grafo LangGraph montado e compilado.")



# ETAPA 4: PONTO DE ENTRADA DO SCRIPT (CLI INTERATIVO)

def main():
    """Esta função contém apenas o loop para rodar o chat no terminal."""
    print("\n🤖 Agente Robbu Orquestrador pronto. Digite \"sair\" para encerrar.")
    messages = []
    while True:
        try:
            user_input = input("Você: ")
            if user_input.strip().lower() == 'sair':
                break
            
            messages.append(HumanMessage(content=user_input))
            out = app.invoke({"messages": messages})
            
            final_response = next(
                (msg for msg in reversed(out["messages"]) if isinstance(msg, AIMessage) and getattr(msg, "name", None) != "classification_decision"),
                AIMessage(content="Desculpe, ocorreu um erro.")
            )
            messages.append(final_response)
            print("Agente:", final_response.content)

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o assistente.")
            break
        except Exception as e:
            print(f"\nOcorreu um erro inesperado: {e}")
            if messages and isinstance(messages[-1], HumanMessage):
                messages.pop()
            print("Reiniciando o ciclo de conversa.")

if __name__ == '__main__':
    main()