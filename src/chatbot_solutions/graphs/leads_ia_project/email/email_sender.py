import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List
import glob

def enviar_arquivo_txt_por_email(
    arquivo_txt: str,
    email_destinatario: str = "ciuryo@gmail.com, carlos.nicacio@robbu.global",
    assunto: str = "Resultado do Qualificador de Leads – Cliente",
    email_remetente: Optional[str] = None,
    senha_remetente: Optional[str] = None,
    servidor_smtp: str = "smtp.gmail.com",
    porta_smtp: int = 587,
    excluir_apos_envio: bool = True
):
    """
    Envia um arquivo .txt por email para múltiplos destinatários.
    
    Args:
        arquivo_txt: Caminho para o arquivo .txt
        email_destinatario: String com emails dos destinatários, separados por vírgula
        assunto: Assunto do email
        email_remetente: Email do remetente (se None, pega do .env)
        senha_remetente: Senha do remetente (se None, pega do .env)
        servidor_smtp: Servidor SMTP (padrão Gmail)
        porta_smtp: Porta SMTP
        excluir_apos_envio: Se True, exclui o arquivo após envio bem-sucedido (padrão: True)
    """
    
    # Carrega credenciais do .env se não fornecidas
    if not email_remetente:
        email_remetente = os.getenv('EMAIL_REMETENTE')
    if not senha_remetente:
        senha_remetente = os.getenv('SENHA_EMAIL')
    
    if not email_remetente or not senha_remetente:
        raise ValueError("Email e senha do remetente são obrigatórios. Configure no .env ou passe como parâmetro.")
    
    # Verifica se o arquivo existe
    if not os.path.exists(arquivo_txt):
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_txt}")
    
    # Cria a mensagem
    msg = MIMEMultipart()
    msg['From'] = email_remetente
    msg['To'] = email_destinatario  # O cabeçalho 'To' pode continuar como string para exibição
    msg['Subject'] = assunto
    
    # Corpo do email
    corpo = f"""
Olá equipe,

Em anexo, segue o relatório gerado pelo qualificador de leads referente ao cliente.

Arquivo: {os.path.basename(arquivo_txt)}

Atenciosamente,
Chatbot Qualificador de Leads da Robbu 🤖
    """
    
    msg.attach(MIMEText(corpo, 'plain', 'utf-8'))
    
    # Anexa o arquivo
    with open(arquivo_txt, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    
    encoders.encode_base64(part)
    part.add_header(
        'Content-Disposition',
        f'attachment; filename= {os.path.basename(arquivo_txt)}'
    )
    
    msg.attach(part)
    
    # --- CORREÇÃO APLICADA AQUI ---
    # A função sendmail espera uma LISTA de destinatários, não uma string com vírgulas.
    # Dividimos a string de emails em uma lista.
    lista_destinatarios = [email.strip() for email in email_destinatario.split(',')]
    
    # Envia o email
    try:
        server = smtplib.SMTP(servidor_smtp, porta_smtp)
        server.starttls()
        server.login(email_remetente, senha_remetente)
        text = msg.as_string()
        # Passa a lista de destinatários para a função sendmail
        server.sendmail(email_remetente, lista_destinatarios, text)
        server.quit()
        
        print(f"Email enviado com sucesso para {', '.join(lista_destinatarios)}")
        
        # Exclui o arquivo .txt após envio bem-sucedido (se solicitado)
        if excluir_apos_envio:
            try:
                os.remove(arquivo_txt)
                print(f"Arquivo {arquivo_txt} excluído com sucesso após envio do email.")
            except Exception as delete_error:
                print(f"Aviso: Não foi possível excluir o arquivo {arquivo_txt}: {str(delete_error)}")
        
        return True
        
    except Exception as e:
        print(f"Erro ao enviar email: {str(e)}")
        return False


def enviar_ultimo_resultado_lead():
    """
    Encontra o arquivo .txt mais recente de resultado de lead e envia por email.
    Busca nas pastas: storage/lead_summaries, storage e raiz do projeto.
    """
    import os
    from pathlib import Path
    
    arquivos_lead = []
    
    # Define o diretório raiz do projeto (vai até a raiz independente de onde o script está)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while project_root.parent != project_root and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    # Procura na pasta storage/lead_summaries
    storage_path = project_root / "storage" / "lead_summaries"
    if storage_path.exists():
        arquivos_storage = list(storage_path.glob("lead_result_*.txt"))
        arquivos_lead.extend([str(f) for f in arquivos_storage])
    
    # Procura na pasta storage geral
    storage_general_path = project_root / "storage"
    if storage_general_path.exists():
        arquivos_storage_general = list(storage_general_path.glob("lead_result_*.txt"))
        arquivos_lead.extend([str(f) for f in arquivos_storage_general])
    
    # Fallback para a raiz do projeto
    os.chdir(project_root)
    arquivos_raiz = glob.glob("lead_result_*.txt")
    arquivos_lead.extend([str(project_root / f) for f in arquivos_raiz])
    
    if not arquivos_lead:
        print("Nenhum arquivo de resultado de lead encontrado.")
        print("Pastas verificadas:")
        print(f"  - {storage_path}")
        print(f"  - {storage_general_path}")
        print(f"  - Raiz do projeto: {project_root}")
        return False
    
    # Pega o arquivo mais recente
    arquivo_mais_recente = max(arquivos_lead, key=os.path.getctime)
    
    print(f"Enviando arquivo: {arquivo_mais_recente}")
    return enviar_arquivo_txt_por_email(arquivo_mais_recente)


def limpar_arquivos_txt_antigos(dias_limite: int = 7):
    """
    Remove arquivos .txt de resultado de leads que são mais antigos que o limite especificado.
    
    Args:
        dias_limite: Número de dias. Arquivos mais antigos que isso serão removidos (padrão: 7 dias)
    """
    import time
    from pathlib import Path
    
    # Define o diretório raiz do projeto
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while project_root.parent != project_root and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    # Lista de diretórios para verificar
    diretorios = [
        project_root / "storage" / "lead_summaries",
        project_root / "storage",
        project_root
    ]
    
    arquivos_removidos = 0
    tempo_limite = time.time() - (dias_limite * 24 * 60 * 60)  # Converte dias para segundos
    
    for diretorio in diretorios:
        if diretorio.exists():
            for arquivo in diretorio.glob("lead_result_*.txt"):
                try:
                    # Verifica se o arquivo é mais antigo que o limite
                    if arquivo.stat().st_mtime < tempo_limite:
                        arquivo.unlink()  # Remove o arquivo
                        print(f"Arquivo antigo removido: {arquivo}")
                        arquivos_removidos += 1
                except Exception as e:
                    print(f"Erro ao remover arquivo {arquivo}: {str(e)}")
    
    print(f"Limpeza concluída. {arquivos_removidos} arquivo(s) removido(s).")
    return arquivos_removidos


def excluir_todos_arquivos_txt():
    """
    Remove TODOS os arquivos .txt de resultado de leads.
    Use com cuidado! Esta função remove todos os arquivos, independente da data.
    """
    from pathlib import Path
    
    # Define o diretório raiz do projeto
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while project_root.parent != project_root and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    # Lista de diretórios para verificar
    diretorios = [
        project_root / "storage" / "lead_summaries",
        project_root / "storage",
        project_root
    ]
    
    arquivos_removidos = 0
    
    for diretorio in diretorios:
        if diretorio.exists():
            for arquivo in diretorio.glob("lead_result_*.txt"):
                try:
                    arquivo.unlink()  # Remove o arquivo
                    print(f"Arquivo removido: {arquivo}")
                    arquivos_removidos += 1
                except Exception as e:
                    print(f"Erro ao remover arquivo {arquivo}: {str(e)}")
    
    print(f"Limpeza total concluída. {arquivos_removidos} arquivo(s) removido(s).")
    return arquivos_removidos


if __name__ == "__main__":
    # Exemplo de uso
    # Para testar, certifique-se de que suas variáveis de ambiente
    # EMAIL_REMETENTE e SENHA_EMAIL estão configuradas.
    
    # Envia o último resultado
    enviar_ultimo_resultado_lead()
    
    # Opcional: limpa arquivos antigos (mais de 7 dias)
    # limpar_arquivos_txt_antigos(7)
    
    # Opcional: remove TODOS os arquivos .txt (use com cuidado!)
    # excluir_todos_arquivos_txt()
