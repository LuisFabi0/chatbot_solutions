from crewai.tools import BaseTool
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import re, requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from ..rd_station.utils import ROBBU_DOCS_CONTEXT
from ..llm.llm import llm
from langchain_core.messages import BaseMessage
class AgentState(TypedDict):
    messages: List[BaseMessage]

# CREW COMO FERRAMENTAS ESPECIALIZADAS
class ContextSearchTool(BaseTool):
    name: str = "Busca na Documentação Interna"
    description: str = "Busca a URL da documentação mais relevante para uma pergunta técnica."

    def _find_best_match(self, query: str, docs_context: List[Dict]) -> Tuple[Dict, int]:
        query_lower = query.lower()
        best_match_doc = None
        best_score = 0
        query_tokens = set(re.findall(r"\w+", query_lower))
        for doc in docs_context:
            score = 0
            doc_name_lower = doc["name"].lower()
            doc_keywords = {k.lower() for k in doc.get("keywords", [])}
            if any(tok in doc_name_lower for tok in query_tokens):
                score += 5
            score += len(query_tokens.intersection(doc_keywords)) * 3
            if score > best_score:
                best_score = score
                best_match_doc = doc
        return best_match_doc, best_score

    def _run(self, query: str) -> str:
        try:
            robbu_match, robbu_score = self._find_best_match(query, ROBBU_DOCS_CONTEXT)
            if robbu_match:
                return robbu_match.get("url", "")
            return "Nenhuma URL encontrada na base local."
        except Exception as e:
            return f"[ERRO_BUSCA_DOCS:{str(e)}]"

class EnhancedWebScrapeTool(BaseTool):
    name: str = "Extração Avançada de Conteúdo Web"
    description: str = "Extrai conteúdo de páginas web com tratamento de erros e formatação."

    def _run(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            main_content = (soup.find('main') or soup.find('article') or soup.find('body'))
            text = main_content.get_text("\n", strip=True) if main_content else ""
            return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])[:4000]
        except Exception as e:
            return f"[ERRO_EXTRACAO:{str(e)}]"

# - A LÓGICA DA CREW ENCAPSULADA EM UMA CLASSE -
class TechnicalCrewExecutor:
    def run(self, query: str) -> str:
        """Executa a Crew de agentes para encontrar uma resposta técnica."""
        pesquisador = Agent(
            role="Especialista em Pesquisa de Documentação",
            goal="Localizar a URL mais relevante na base de conhecimento para responder a uma pergunta técnica.",
            backstory="Você é um especialista em encontrar a fonte de informação correta dentro de uma base de conhecimento pré-definida para responder a pergunta do usuário.",
            llm=llm, tools=[ContextSearchTool()], verbose=True
        )
        extrator = Agent(
            role="Extrator de Conteúdo Web",
            goal="Extrair o conteúdo essencial de uma página web de forma limpa e objetiva.",
            backstory="Especialista em parsing de HTML, focado em extrair apenas o texto relevante de uma página.",
            llm=llm, tools=[EnhancedWebScrapeTool()], verbose=True
        )
        redator = Agent(
            role="Redator Técnico Profissional",
            goal="Produzir respostas claras, objetivas e profissionais baseadas em conteúdo técnico. Você transforma jargão técnico em respostas fáceis de entender formatadas de forma personalizada para o usuário final.",
            backstory="Você é um especialista em suporte técnico que se comunica de forma concisa e direta, sempre citando a fonte oficial.",
            llm=llm, verbose=True
        )

        tarefa_pesquisa = Task(description=f"Encontre a URL da documentação para: '{query}'", expected_output="A URL mais relevante.", agent=pesquisador)

        crew_pesquisa = Crew(agents=[pesquisador], tasks=[tarefa_pesquisa], process=Process.sequential, verbose=True)
        url = crew_pesquisa.kickoff()

        if not str(url).startswith("http"):
            return "Não localizei uma página específica para essa dúvida. Você pode detalhar o erro, código ou endpoint?"

        tarefa_extracao = Task(description=f"Extraia o conteúdo da URL: {url}", expected_output="Texto limpo e objetivo.", agent=extrator)
        tarefa_redacao = Task(
            description=f"Produza uma resposta técnica objetiva para a pergunta '{query}', usando o conteúdo extraído. Sua resposta deve ser em português, profissional, e terminar com a fonte: {url}",
            expected_output="Resposta técnica profissional.",
            agent=redator,
            context=[tarefa_extracao]
        )

        crew_completa = Crew(agents=[extrator, redator], tasks=[tarefa_extracao, tarefa_redacao], process=Process.sequential, verbose=True)
        resultado_final = crew_completa.kickoff()

        return resultado_final if resultado_final else f"Não foi possível processar a solicitação para a URL: {url}"

