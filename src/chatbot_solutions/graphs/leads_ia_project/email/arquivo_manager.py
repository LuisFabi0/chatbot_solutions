"""
Utilitário para gerenciar arquivos .txt de resultado de leads.
Este módulo fornece funções para enviar emails e gerenciar a exclusão automática dos arquivos.
"""
import os
from pathlib import Path

# Import direto do email_sender na mesma pasta
from email_sender import (
    enviar_arquivo_txt_por_email,
    enviar_ultimo_resultado_lead,
    limpar_arquivos_txt_antigos,
    excluir_todos_arquivos_txt
)


def enviar_e_excluir_arquivo(arquivo_txt: str, **kwargs):
    """
    Envia um arquivo .txt específico por email e o exclui automaticamente.
    
    Args:
        arquivo_txt: Caminho para o arquivo .txt
        **kwargs: Argumentos adicionais para a função de envio de email
    
    Returns:
        bool: True se o envio e exclusão foram bem-sucedidos
    """
    # Por padrão, sempre exclui após o envio
    kwargs.setdefault('excluir_apos_envio', True)
    
    return enviar_arquivo_txt_por_email(arquivo_txt, **kwargs)


def processar_todos_arquivos_pendentes():
    """
    Encontra todos os arquivos .txt de leads pendentes e os envia por email,
    excluindo-os após o envio bem-sucedido.
    
    Returns:
        dict: Estatísticas do processamento
    """
    from pathlib import Path
    
    # Define o diretório raiz do projeto
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while project_root.parent != project_root and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    arquivos_lead = []
    
    # Procura em todas as pastas relevantes
    diretorios = [
        project_root / "storage" / "lead_summaries",
        project_root / "storage",
        project_root
    ]
    
    for diretorio in diretorios:
        if diretorio.exists():
            arquivos_encontrados = list(diretorio.glob("lead_result_*.txt"))
            arquivos_lead.extend([str(f) for f in arquivos_encontrados])
    
    estatisticas = {
        'total_encontrados': len(arquivos_lead),
        'enviados_com_sucesso': 0,
        'falhas_envio': 0,
        'arquivos_processados': []
    }
    
    print(f"Encontrados {len(arquivos_lead)} arquivo(s) para processar.")
    
    for arquivo in arquivos_lead:
        print(f"\nProcessando: {arquivo}")
        try:
            sucesso = enviar_e_excluir_arquivo(arquivo)
            if sucesso:
                estatisticas['enviados_com_sucesso'] += 1
                estatisticas['arquivos_processados'].append({
                    'arquivo': arquivo,
                    'status': 'sucesso'
                })
            else:
                estatisticas['falhas_envio'] += 1
                estatisticas['arquivos_processados'].append({
                    'arquivo': arquivo,
                    'status': 'falha_envio'
                })
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {str(e)}")
            estatisticas['falhas_envio'] += 1
            estatisticas['arquivos_processados'].append({
                'arquivo': arquivo,
                'status': 'erro',
                'erro': str(e)
            })
    
    print(f"\n=== RESUMO DO PROCESSAMENTO ===")
    print(f"Total de arquivos encontrados: {estatisticas['total_encontrados']}")
    print(f"Enviados com sucesso: {estatisticas['enviados_com_sucesso']}")
    print(f"Falhas no envio: {estatisticas['falhas_envio']}")
    
    return estatisticas


def configurar_limpeza_automatica():
    """
    Configura um script para limpeza automática de arquivos antigos.
    Cria um arquivo .bat (Windows) ou .sh (Linux/Mac) para executar a limpeza.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while project_root.parent != project_root and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    # Script para Windows
    bat_content = f"""@echo off
echo Executando limpeza automatica de arquivos .txt antigos...
cd /d "{project_root}"
python -c "from src.chatbot_solutions.graphs.leads_ia_project.email.email_sender import limpar_arquivos_txt_antigos; limpar_arquivos_txt_antigos(7)"
echo Limpeza concluida.
pause
"""
    
    bat_file = project_root / "limpeza_automatica.bat"
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    # Script para Linux/Mac
    sh_content = f"""#!/bin/bash
echo "Executando limpeza automática de arquivos .txt antigos..."
cd "{project_root}"
python3 -c "from src.chatbot_solutions.graphs.leads_ia_project.email.email_sender import limpar_arquivos_txt_antigos; limpar_arquivos_txt_antigos(7)"
echo "Limpeza concluída."
"""
    
    sh_file = project_root / "limpeza_automatica.sh"
    with open(sh_file, 'w', encoding='utf-8') as f:
        f.write(sh_content)
    
    # Torna o script executável no Linux/Mac
    try:
        os.chmod(sh_file, 0o755)
    except:
        pass  # Ignora erro no Windows
    
    print(f"Scripts de limpeza automática criados:")
    print(f"  Windows: {bat_file}")
    print(f"  Linux/Mac: {sh_file}")
    print(f"\nPara executar a limpeza automática:")
    print(f"  Windows: Execute o arquivo {bat_file.name}")
    print(f"  Linux/Mac: Execute ./limpeza_automatica.sh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerenciador de arquivos .txt de leads")
    parser.add_argument('--processar-todos', action='store_true', 
                       help='Processa todos os arquivos .txt pendentes')
    parser.add_argument('--limpar-antigos', type=int, metavar='DIAS', 
                       help='Remove arquivos mais antigos que X dias')
    parser.add_argument('--excluir-todos', action='store_true', 
                       help='Remove TODOS os arquivos .txt (use com cuidado!)')
    parser.add_argument('--configurar-limpeza', action='store_true', 
                       help='Cria scripts para limpeza automática')
    parser.add_argument('--enviar-ultimo', action='store_true', 
                       help='Envia apenas o arquivo mais recente')
    
    args = parser.parse_args()
    
    if args.processar_todos:
        processar_todos_arquivos_pendentes()
    elif args.limpar_antigos:
        limpar_arquivos_txt_antigos(args.limpar_antigos)
    elif args.excluir_todos:
        resposta = input("ATENÇÃO: Isso irá remover TODOS os arquivos .txt. Confirma? (digite 'SIM'): ")
        if resposta == 'SIM':
            excluir_todos_arquivos_txt()
        else:
            print("Operação cancelada.")
    elif args.configurar_limpeza:
        configurar_limpeza_automatica()
    elif args.enviar_ultimo:
        enviar_ultimo_resultado_lead()
    else:
        print("Uso: python arquivo_manager.py [opção]")
        print("Use --help para ver todas as opções disponíveis.")
