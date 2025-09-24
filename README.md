# 🤖 IA para Classificação de Imagens com PyTorch

Este projeto é uma **inteligência artificial para classificação de imagens**, construída com uma rede neural em PyTorch. O sistema é flexível e pode ser treinado para classificar qualquer tipo de imagem, sendo capaz de aprender, validar e classificar novos dados. A implementação inclui um mecanismo de *early stopping* para otimizar o processo de treinamento e evitar overfitting.

## ✨ Principais Funcionalidades

- **IA Generalista**: Projetado para classificar qualquer dataset de imagens, não apenas um tipo específico.
- **Rede Neural Flexível**: Implementação de uma rede totalmente conectada (fully connected) para classificação.
- **Pré-processamento de Imagens**: Redimensionamento e conversão automática das imagens para tensores PyTorch.
- **Treinamento Otimizado**: Inclui *early stopping* para interromper o treinamento quando a acurácia de validação para de melhorar.
- **Classificação de Novas Imagens**: Função para classificar imagens individuais e visualizar os resultados com `matplotlib`.
- **Configuração Centralizada**: Todos os hiperparâmetros são facilmente ajustáveis em um único arquivo de configuração.

## 📂 Estrutura de Pastas

Você pode usar **qualquer banco de dados de imagens** de sua escolha. O único requisito é organizar as imagens em pastas de `train` (treino) e `test` (teste), com subpastas para cada classe que você deseja classificar.

Veja a estrutura de exemplo abaixo:

```
/
├── PyTorchRN.py           # Script principal (treino, validação e classificação)
├── Config_Parametros.py   # Configurações de hiperparâmetros
├── Carregar_Banco.py      # Lógica para carregar os datasets
└── seu_banco_de_imagens/
    ├── train/
    │   ├── classe_A/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── classe_B/
    │       ├── imgA.jpg
    │       └── ...
    └── test/
        ├── classe_A/
        │   ├── img_val_1.jpg
        │   └── ...
        └── classe_B/
            ├── img_val_A.jpg
            └── ...
```
*Substitua `seu_banco_de_imagens`, `classe_A` e `classe_B` pelos nomes do seu dataset e das suas categorias.*

## 🚀 Começando

Siga os passos abaixo para executar o projeto localmente.

### Pré-requisitos

- Python 3.8 ou superior

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Instale as dependências:**
    ```bash
    pip install torch torchvision matplotlib scikit-learn tensorboard
    ```

### Configuração

Ajuste os hiperparâmetros e as configurações do modelo no arquivo `Config_Parametros.py`:

```python
# Config_Parametros.py

epocas = 100             # Número máximo de épocas
tamanho_lote = 33        # Tamanho do batch (batch size)
taxa_aprendizagem = 0.001 # Taxa de aprendizado (learning rate)
tamanho_imagens = 64     # Redimensionar imagens para 64x64 pixels
perc_val = 0.21          # Percentual de imagens de treino para validação interna
paciencia = 5            # Épocas sem melhora antes de parar (early stopping)
tolerancia = 0.01        # Melhora mínima na acurácia para considerar progresso
nome_rede = "meu_modelo" # Nome base para salvar o modelo treinado
```

## ▶️ Como Executar

Para treinar o modelo com seu dataset, execute o script principal:

```bash
python PyTorchRN.py
```

O script irá realizar as seguintes ações:
1.  Carregar e pré-processar as imagens das pastas `train` e `test`.
2.  Iniciar o ciclo de treinamento e validação.
3.  Aplicar o *early stopping* se a acurácia de validação não melhorar.
4.  Salvar o modelo com melhor desempenho como `modelo_treinado.pth`.
5.  Carregar o modelo salvo e classificar imagens aleatórias do conjunto de teste, exibindo as previsões.

## 💡 Observações Importantes

-   O modelo atual é uma rede simples, totalmente conectada. Para datasets mais complexos ou imagens de alta resolução, considere o uso de **Redes Neurais Convolucionais (CNNs)** para obter melhores resultados.
-   Se você adicionar a normalização de imagens (`transforms.Normalize`) ao seu pipeline, lembre-se de **desnormalizar** os tensores antes de exibi-los com `matplotlib`.
