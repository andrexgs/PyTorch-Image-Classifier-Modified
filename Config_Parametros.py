"""
    Arquivo para configuração da RN, baseado no script exemplo_pytorch_v4.
"""

# Hiperparâmetros de treinamento
epocas = 100              # Total de passagens pelo conjunto de imagens
tamanho_lote = 16         # Tamanho de cada lote (batch)
taxa_aprendizagem = 0.02   # Magnitude das alterações nos pesos (ajustado para Adam)
momento = 0.9               # Parâmetro para o otimizador SGD (se for usado)

# Hiperparâmetros de Parada Antecipada (Early Stopping)
paciencia = 5             # Total de épocas sem melhoria até parar
tolerancia = 0.9         # Melhoria menor que este valor não é considerada

# Configuração da Arquitetura
# Opções: "resnet", "squeezenet", "densenet"
nome_rede = "resnet"
tamanho_imagens = 64     # Tamanho de imagem ideal para essas arquiteturas
perc_val = 0.21             # Percentual do treino a ser usado para validação