import asyncio
import json
import os

import aioconsole
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from chat_service import chat
from vector_builder import (cria_banco_vetorial, save_vectorstore_qdrant,
                            save_vectorstore_qdrant_incremental)
from vector_builder_confluence import processa_dados_confluence
from vector_builder_youtube import processa_dados_youtube

app = FastAPI()

# rotina para testar o serviço
async def test():
  while True:
      print("\n")
      pergunta = input("O que você quer saber? ")
      if pergunta == ".":
        break
      resposta = await chat(pergunta)
      print("\nResposta:\n\n"+resposta)



# criaria um end-point para recriar o banco vetorial com chave especial
@app.post("/cria_banco_vetorial/")
async def cria_banco_vetorial():
    total_chunks = []

    chunks = await processa_dados_youtube()
    for chunk in chunks:
        total_chunks.append(chunk)

    chunks = await processa_dados_confluence()
    for chunk in chunks:
        total_chunks.append(chunk)

    await save_vectorstore_qdrant(total_chunks)
    # teste:
    await save_vectorstore_qdrant_incremental(total_chunks)

    
async def main():
  load_dotenv()
  criacao_base_vetorial_permitido = os.getenv("CRIACAO_BASE_VETORIAL_PERMITIDA", "N")

  print("Linx Assistente de IA")
  print("1-Chat")
  if criacao_base_vetorial_permitido == "S":
    print("2-Criar base vetorial")
  opcao = input("Escolha a opção:")

  if opcao == "1":
    print("Iniciando o serviço de chat via console.")
    await test()
  elif opcao == "2":
    if criacao_base_vetorial_permitido != "S":
      print("Criação da base vetorial não permitida.")
      return
    confirma = input("Confirma criação da base vetorial? (SIM/N):")
    if confirma == "S":
      #########################################
      await cria_banco_vetorial()
      ########################################
    else:
      print("Criação da Base vetorial foi cancelada.")


if __name__ == "__main__":
  #tirar essa
  asyncio.run(main())

  #colocar essa
  #uvicorn.run("main:app", host="0.0.0.0", port=8107)

