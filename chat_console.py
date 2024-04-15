import asyncio
import json
import os

import aioconsole
import uvicorn
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from chat_service import chat
from vector_builder import cria_banco_vetorial


# rotina para testar o serviço
async def test():
  while True:
      print("\n")
      pergunta = input("O que você quer saber? ")
      if pergunta == ".":
        break
      resposta = await chat(pergunta)
      print("\nResposta:\n\n"+resposta)


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
  asyncio.run(main())

