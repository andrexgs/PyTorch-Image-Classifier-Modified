"""
    IMPORTANDO AS BIBLIOTECAS PARA REDE NEURAL DE CLASSIFICAÇÃO
"""
import torch   # Biblioteca pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
from torch.utils.data import DataLoader # Manipulação de bancos de imagens
from torchvision import datasets, models # Ajuda a importar alguns bancos já prontos e famosos
from torchvision.transforms import ToTensor # Realiza transformações nas imagens
import matplotlib.pyplot as plt # Mostra imagens e gráficos
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem
import math
import Config_Parametros
import Carregar_Banco


"""
    TRANSFORMANDO IMAGENS EM TENSORES E NORMALIZANDO
"""
transform = transforms.Compose([
    transforms.Resize((Config_Parametros.tamanho_imagens,Config_Parametros.tamanho_imagens)),
    transforms.ToTensor(),])


"""
    CARREGANDO O BANCO DE IMAGENS 
"""
training_val_data = datasets.ImageFolder(root=Carregar_Banco.pasta_treino,transform=transform) 
test_data = datasets.ImageFolder(root=Carregar_Banco.pasta_validacao,transform=transform)
train_idx, val_idx = train_test_split(list(range(len(training_val_data))), test_size=Config_Parametros.perc_val) # Gera uma lista com os índices de imagens
training_data = Subset(training_val_data, train_idx) # Cria um dataset somente com índice de treino
val_data = Subset(training_val_data, val_idx) # Cria um dataset somente com índice de validação


"""
    CRIANDO OS LOTES (BATCHES)
"""
train_dataloader = DataLoader(training_data, batch_size=Config_Parametros.tamanho_lote,shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=Config_Parametros.tamanho_lote,shuffle=True)


"""
    INFORMAÇÕES GERAIS
    
        shape[0] -> Quantidade de imagens em cada lote      shape[1] -> Quantidade de canais (1 para Preto e Branco | 3 para RGB)
        shape[2] -> Altura de cada imagem                   shape[3] -> Largura de Cada imagem
        dtype -> Tipo da classe
"""
for X, y in val_dataloader: # X-> Tensor de imagens do lote  y-> Tensor com as classes
    print(f"\nTamanho do lote de imagens: {X.shape[0]}")
    print(f"Quantidade de canais: {X.shape[1]}")
    print(f"Altura de cada imagem: {X.shape[2]}")
    print(f"Largura de cada imagem: {X.shape[3]}")
    print(f"Tamanho do lote de classes (labels): {y.shape[0]}")
    print(f"Tipo de cada classe: {y.dtype}")
    break  # Para após mostrar os dados do 1º lote

print(f"\nTotal de imagens de treinamento: {len(training_data)}")
print(f"Total de imagens de validação: {len(val_data)}")
labels_map = {v: k for k, v in test_data.class_to_idx.items()} # Dicionário que relaciona um nome das classes (labels) em um número
print(f'\nClasses:',labels_map)


"""
    VERIFICA SE A MÁQUINA ESTÁ USANDO A GPU OU CPU
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nUsando {device}")

total_classes = len(labels_map) # Calcula o tamanho de classes (labels) possui 
tamanho_entrada_flatten = Config_Parametros.tamanho_imagens * Config_Parametros.tamanho_imagens * 3 # Tamanho vetorial de cada imagem


"""
    DEFININDO A CLASSE DA RN
"""
class NeuralNetwork(nn.Module):
    def __init__(self):


        super(NeuralNetwork, self).__init__() # Inicializa a classe principal
        self.flatten = nn.Flatten() # Transforma o redimencionamento das imagens em um vetor

        # Neurônios
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(tamanho_entrada_flatten, 512), # Transforma o vetor da imagem em um vetor de 512 neurônios
            nn.ReLU(),  
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, total_classes)) # Transforma o vetor de 512 neurônios em um vetor com a quantidade de classes (labels)

    # Define os dados passam pela rede
    def forward(self, x):
        x = self.flatten(x) # Faz o achatamento das imagens
        output_values = self.linear_relu_stack(x) # Ativação dos neurônios
        return output_values

model = NeuralNetwork().to(device) # Cria a rede e joga na GPU ou CPU
print(model) # Impreme os dados da arquitetura de rede


# Pega todos os parâmetros da rede e controla o tamanho dos passos "Aprendizagem"
otimizador = torch.optim.Adam(model.parameters(), lr=Config_Parametros.taxa_aprendizagem) 
# Mede o erro entre as previsões da rede e as classes reais
funcao_perda = nn.CrossEntropyLoss() 

"""
    DEFININDO A FUNÇÃO DE TREINAMENTO DA RN
"""

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader, start=1):

        X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
        pred = model(X)         # Realiza uma previsão usando os pesos atuais
        loss = loss_fn(pred, y) # Calcula o erro com os pesos atuais

        optimizer.zero_grad()  # Zera os gradientes 
        loss.backward()        # Retropropaga o gradiente do erro
        optimizer.step()       # Recalcula todos os pesos da rede

        loss, current = loss.item(), min(batch * dataloader.batch_size, len(dataloader.dataset))
        print(f"Perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")


"""
    DEFININDO A FUNÇÃO DE VALIDAÇÃO DA RN
"""
def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total de imagens para validação
    num_batches = len(dataloader)   # Total de lotes (batches)
    model.eval()  # Coloca a rede em modo de avaliação (e não de aprendizagem)
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # Prepara os dados para o dispositivo (GPU ou CPU)
            pred = model(X) # Realiza uma previsão usando os pesos atuais
            val_loss += loss_fn(pred, y).item() # Soma da perda do lote (batch) atual
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Soma de quantos acertos

    val_loss /= num_batches # Média de perda nos lotes
    acuracia = correct / size # Acertos no conjunto de validação  
    
    print("\n..::: INFORMAÇÕES DE VALIDAÇÃO :::..\n")
    print(f"    Total de acertos: {int(correct)}")
    print(f"    Total de imagens: {size}")
    print(f"    Perda média: {val_loss:>8f}")            
    print(f"    Acurácia: {(acuracia):>0.3f}")
    print(f"    Acurácia percentual: {(100*acuracia):>0.1f}%")
    return acuracia

melhor_acuracia = -math.inf  # Começamos com um valor muito baixo (infinito negativo)
total_sem_melhora = 0

# Loop que passa todas as imagens várias vezes pela quantidade de épocas
for t in range(Config_Parametros.epocas):
    print(f"-------------------------------")
    print(f"Época {t+1}\n-------------------------------")
    train(train_dataloader, model, funcao_perda, otimizador)
    acuracia_val = validation(val_dataloader, model, funcao_perda)

    # LÓGICA CORRIGIDA: Compara a acurácia atual com a melhor já registrada
    if acuracia_val > melhor_acuracia:
        print(f"\n>>> Acurácia melhorou ({melhor_acuracia:.3f} --> {acuracia_val:.3f}). Salvando modelo...")
        melhor_acuracia = acuracia_val  # Atualiza a melhor acurácia
        torch.save(model.state_dict(), "modelo_treinado.pth") # Salva o modelo
        total_sem_melhora = 0 # Reseta o contador de paciência
    else:
        total_sem_melhora += 1
        print(f"\n>>> Acurácia não melhorou. Paciência: {total_sem_melhora}/{Config_Parametros.paciencia}")

    # Condição de parada antecipada
    if total_sem_melhora >= Config_Parametros.paciencia:
        print(f"\nParada antecipada na época {t+1}! O modelo não melhora há {Config_Parametros.paciencia} épocas.")
        break

print("Terminou a fase de aprendizagem !")

"""
    CARREGA A REDE NEURAL TREINADA ANTERIORMENTE
"""
model = NeuralNetwork().to(device) # Cria uma nova instância do modelo na GPU/CPU
model.load_state_dict(torch.load("modelo_treinado.pth"))

"""
    CLASSIFICA UMA IMAGEM
"""
def classifica_uma_imagem(model, x, y):
    model.eval()
    with torch.no_grad():
       # Adicione a linha abaixo para mover o tensor da imagem para a GPU/CPU
       x = x.to(device)
       
       x = x.unsqueeze(0)
       pred = model(x)
       predita, real = labels_map[int(pred[0].argmax(0))], labels_map[y]
       print(f'Predita: "{predita}", Real: "{real}"')
    return predita

figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 4
print(f"-------------------------------")
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # Número randomico menor que a quantidade de imagens
    img, label = training_data[sample_idx] # Pega a imagem e classificação usando o número randomico
    predita = classifica_uma_imagem(model,img,label) # Classifica usando a rede treinada
    figure.add_subplot(rows, cols, i) # Adiciona a imagem na grade que será mostrada
    plt.title(predita) # Usa a classe da imagem como título da imagem
    plt.axis("off") # Não mostra valores nos eixos X e Y
    plt.imshow(img.permute(1,2,0)) # Ajusta a ordem das dimensões do tensor
print(f"-------------------------------")
plt.show() # Este é o comando que vai mostrar as imagens
