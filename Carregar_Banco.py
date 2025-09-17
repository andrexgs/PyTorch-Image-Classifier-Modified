"""
    Arquivo para configuração e carregamento do banco de imagens a partir do Google Drive.
"""
import gdown
import os
import shutil

# --- CONFIGURAÇÃO ---
# 1. Cole o link da sua pasta COMPARTILHADA do Google Drive aqui.
#    Lembre-se: em "Acesso geral", deve estar como "Qualquer pessoa com o link".

google_drive_url = "https://drive.google.com/drive/folders/1UxUpQXu-y1FFEnrcNfAEWz0QZiYG-m2l?usp=sharing"

# 2. Nome da pasta que será criada localmente para armazenar as imagens.
#    É esta pasta que você deve apagar se precisar baixar os dados novamente.
pasta_banco_imagens = "bd_imagens_drive"

# --- LÓGICA DE DOWNLOAD ---

# IMPORTANTE: Se você atualizar sua pasta no Google Drive (adicionar/remover imagens),
# apague a pasta "bd_imagens_drive" do seu computador manualmente para forçar
# o download da nova versão na próxima vez que executar o script.

# Verifica se a pasta de imagens já foi baixada
if not os.path.exists(pasta_banco_imagens):
    print(f"A pasta '{pasta_banco_imagens}' não foi encontrada localmente.")
    print(f"Baixando o banco de imagens de: {google_drive_url}")
    
    try:
        # Faz o download da pasta do Google Drive
        gdown.download_folder(google_drive_url, output=pasta_banco_imagens, quiet=False, use_cookies=False)
        print("\nDownload concluído com sucesso!")
    except Exception as e:
        print(f"\nOcorreu um erro durante o download: {e}")
        print("Verifique se o link do Google Drive está correto e se as permissões de compartilhamento estão como 'Qualquer pessoa com o link'.")
        # Se o download falhar, remove a pasta parcialmente criada para evitar erros futuros.
        if os.path.exists(pasta_banco_imagens):
            shutil.rmtree(pasta_banco_imagens)
        exit() # Encerra o script se o download falhar
else:
    print(f"O banco de imagens '{pasta_banco_imagens}' já existe localmente. Usando a versão em cache.")
    print("(Para baixar novamente, apague a pasta localmente e rode o script de novo.)")


# --- DEFINIÇÃO DOS CAMINHOS PARA A REDE NEURAL ---

# Define os caminhos que serão usados pelo PyTorchRN.py para encontrar as imagens de treino e validação.
pasta_raiz = "./"
pasta_treino = os.path.join(pasta_raiz, pasta_banco_imagens, "train")
pasta_validacao = os.path.join(pasta_raiz, pasta_banco_imagens, "test")

# Verifica se os caminhos de treino e validação existem antes de prosseguir
if not os.path.exists(pasta_treino) or not os.path.exists(pasta_validacao):
    print("\nERRO: As pastas 'train' e 'test' não foram encontradas dentro do diretório baixado.")
    print(f"Verifique a estrutura da sua pasta no Google Drive. Ela deve conter subpastas chamadas 'train' e 'test'.")
    exit()

print(f"\nLeitura das imagens configurada:")
print(f"  - Pasta de treino:    {pasta_treino}")
print(f"  - Pasta de validação: {pasta_validacao}")