"""
    Arquivo para configuração da RN
"""

epocas = 100  # Quantidade de vezes que a imagem irá passar pelo conjunto de treinamento
tamanho_lote = 33  # Tamanho de cada lote (batches) que é um pequeno grupo de imagens
taxa_aprendizagem = 0.001   # Magnitude nos pesos

nome_rede = "resnet" # Define uma arquitetura 
tamanho_imagens = 64 
perc_val = 0.21  # Percentual do treinamento a ser usado para validação

paciencia = 5  # Total de épocas sem melhoria da acurácia na validação até parar
tolerancia = 0.01 # Melhoria menor que este valor não é considerada melhoria