import os
import json
import asyncio
import aioconsole
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
import uvicorn
from chat_service import chat
from llm import prepara_llm, prepara_llm_azure
from prompt import prepara_prompt
from vector_builder import cria_banco_vetorial
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


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
  print("Linx Assistente de IA")
  print("1-Chat")
  print("2-Criar base vetorial")
  print("Escolha a opção:")
  opcao = input("Escolha a opção:")

  if opcao == "1":
    print("Iniciando o serviço de chat via console.")
    await test()
  elif opcao == "2":
    criacao_base_vetorial_permitido = os.getenv("CRIACAO_BASE_VETORIAL_PERMITIDA", "N")
    if criacao_base_vetorial_permitido != "S":
      print("Criação da base vetorial não permitida.")
      return
    confirma = input("Confirma criação da base vetorial? (S/N): ")
    if confirma == "S":
      await cria_banco_vetorial()
    else:
      print("Criação da Base vetorial foi cancelada.")


if __name__ == "__main__":
  asyncio.run(main())

