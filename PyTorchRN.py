# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import math
import numpy as np
from PIL import Image

# Importa as configurações
import Config_Parametros as cfg
import Carregar_Banco

# --- CONFIGURAÇÕES GLOBAIS ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define as transformações nas imagens, incluindo Data Augmentation para o treino
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(cfg.tamanho_imagens),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(int(cfg.tamanho_imagens * 1.14)), # 256 para 224
        transforms.CenterCrop(cfg.tamanho_imagens),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- CARREGAMENTO DE DADOS ---
def carregar_dados():
    full_train_dataset = datasets.ImageFolder(root=Carregar_Banco.pasta_treino, transform=data_transforms['train'])
    full_val_dataset = datasets.ImageFolder(root=Carregar_Banco.pasta_treino, transform=data_transforms['val'])
    test_data = datasets.ImageFolder(root=Carregar_Banco.pasta_validacao, transform=data_transforms['val'])

    train_idx, val_idx = train_test_split(list(range(len(full_train_dataset))), test_size=cfg.perc_val)
    training_data = Subset(full_train_dataset, train_idx)
    val_data = Subset(full_val_dataset, val_idx)

    train_dl = DataLoader(training_data, batch_size=cfg.tamanho_lote, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=cfg.tamanho_lote, shuffle=True)
    
    labels_map = {v: k for k, v in test_data.class_to_idx.items()}
    return train_dl, val_dl, test_data, labels_map

# --- MODELO ---
def carregar_modelo(num_classes):
    model = None
    if cfg.nome_rede == "resnet":
        model = models.resnet18(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif cfg.nome_rede == "squeezenet":
        model = models.squeezenet1_0(weights='SQUEEZENET1_0_IMAGENET1K_V1')
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif cfg.nome_rede == "densenet":
        model = models.densenet161(weights='DENSENET161_IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model.to(device)

# --- FUNÇÕES DE TREINO E VALIDAÇÃO ---
def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, train_correct = 0, 0
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return train_loss / num_batches, train_correct / size

def validation_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    val_loss, val_correct = 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return val_loss / num_batches, val_correct / size

# --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO ---
async def iniciar_treinamento(update, context):
    train_dl, val_dl, _, labels_map = carregar_dados()
    total_classes = len(labels_map)
    model = carregar_modelo(total_classes)

    funcao_perda = nn.CrossEntropyLoss()
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    otimizador = torch.optim.Adam(params_to_update, lr=cfg.taxa_aprendizagem)

    maior_acuracia = 0
    total_sem_melhora = 0
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Iniciando treinamento com a arquitetura '{cfg.nome_rede}'...")

    for epoca in range(cfg.epocas):
        train_loss, train_acuracia = train_epoch(train_dl, model, funcao_perda, otimizador)
        val_loss, val_acuracia = validation_epoch(val_dl, model, funcao_perda)

        msg = (f"Época {epoca+1}:\n"
               f"  - Acurácia de Treino: {100*train_acuracia:.1f}%\n"
               f"  - Acurácia de Validação: {100*val_acuracia:.1f}%")
        
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

        if val_acuracia > (maior_acuracia + cfg.tolerancia):
            torch.save(model.state_dict(), "modelo_treinado.pth")
            maior_acuracia = val_acuracia
            total_sem_melhora = 0
        else:
            total_sem_melhora += 1

        if total_sem_melhora >= cfg.paciencia:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Parada antecipada na época {epoca+1}!")
            break
            
    return "Treinamento finalizado! Use /test para ver as métricas de desempenho."

# --- FUNÇÕES DE AVALIAÇÃO E CLASSIFICAÇÃO ---
def gerar_metricas_teste():
    _, _, test_data, labels_map = carregar_dados()
    model = carregar_modelo(len(labels_map))
    try:
        model.load_state_dict(torch.load("modelo_treinado.pth"))
    except FileNotFoundError:
        return "Modelo não encontrado. Treine primeiro com /train.", None

    predicoes, reais = [], []
    model.eval()
    with torch.no_grad():
        for img, label in test_data:
            img = img.unsqueeze(0).to(device)
            pred = model(img)
            predicoes.append(int(pred[0].argmax(0)))
            reais.append(label)

    report_str = classification_report(reais, predicoes, target_names=labels_map.values())
    
    # Gerar e salvar matriz de confusão
    matriz = confusion_matrix(reais, predicoes)
    df_matriz = pd.DataFrame(matriz, index=labels_map.values(), columns=labels_map.values())
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_matriz, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    caminho_imagem = "matriz_confusao.png"
    plt.savefig(caminho_imagem)
    plt.close()

    return f"```{report_str}```", caminho_imagem

def classificar_imagem(image_path):
    _, _, _, labels_map = carregar_dados()
    model = carregar_modelo(len(labels_map))
    try:
        model.load_state_dict(torch.load("modelo_treinado.pth"))
    except FileNotFoundError:
        return "Modelo não encontrado. Treine primeiro com /train."

    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = data_transforms['val'](image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        predita = labels_map[int(pred[0].argmax(0))]
    return f"A imagem foi classificada como: **{predita}**"