import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Importa as funções do nosso script de RN
import PyTorchRN as rn

# Configura o logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Funções de Comando para o Bot ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Envia uma mensagem de boas-vindas."""
    user = update.effective_user
    await update.message.reply_html(
        f"Olá, {user.mention_html()}!\n\n"
        "Eu sou um bot classificador de imagens com PyTorch.\n\n"
        "<b>Comandos disponíveis:</b>\n"
        "  - /train : Inicia o treinamento da rede neural.\n"
        "  - /test : Avalia o modelo treinado no conjunto de teste.\n"
        "  - Envie uma imagem para classificá-la (após o treino)."
    )

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inicia o processo de treinamento e envia atualizações."""
    await update.message.reply_text("✅ Comando recebido. O treinamento será iniciado em segundo plano...")
    
    # A função de treinamento agora é assíncrona e envia as mensagens
    resultado_final = await rn.iniciar_treinamento(update, context)
    
    await update.message.reply_text(resultado_final)

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executa o teste no modelo treinado e envia as métricas."""
    await update.message.reply_text("Avaliando o modelo no conjunto de teste...")
    
    report_str, matriz_path = rn.gerar_metricas_teste()
    
    await update.message.reply_markdown_v2(report_str)
    
    if matriz_path:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(matriz_path, 'rb'))
        os.remove(matriz_path)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa e classifica a imagem recebida."""
    photo_file = await update.message.photo[-1].get_file()
    
    file_path = f"temp_{photo_file.file_id}.jpg"
    await photo_file.download_to_drive(file_path)
    
    await update.message.reply_text("Recebi sua imagem. Classificando...")

    prediction_md = rn.classificar_imagem(file_path)
    
    await update.message.reply_markdown(prediction_md)
    
    os.remove(file_path)

def main():
    """Inicia o bot."""
    TOKEN = "7538688960:AAGJjkFnHu0AMLUeGDG6DmhzUVSD68abFgo"
    
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot iniciado... Pressione Ctrl+C para parar.")
    application.run_polling()

if __name__ == '__main__':
    main()