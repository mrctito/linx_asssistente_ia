import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import uvicorn
from llm import prepara_llm, prepara_llm_azure
from prompt import prepara_prompt
from vector_builder_confluence import cria_banco_confluence

app = FastAPI()

# rotina para testar o serviço
def test():
  prompt = prepara_prompt()
  llm = prepara_llm(prompt)
  while True:
      print("\n")
      texto_usuario = input("Digite o comando desejado: ")
      if texto_usuario == ".":
        break

      usuario_input = UsuarioInput(codigo_sistema="EMPORIO", texto_usuario=texto_usuario)
      result = decompoe_acao(usuario_input)
      print(result)


class UsuarioInput(BaseModel):
  codigo_sistema: str
  texto_usuario: str

# serviço que recebe um comando do usuário e retorna o código de menu correspondente
@app.post("/decompoe_acao/")
def decompoe_acao(usuario_input: UsuarioInput) -> str:
  prompt = prepara_prompt()

  # cria LLMChain nativo OpenAI ou Azure
  if os.getenv("USE_AZURE", "N") == "S":
    llm = prepara_llm_azure(prompt)
  else:
    llm = prepara_llm(prompt)
  
  # chama a API que traduz o comando do usuario em um código de menu
  result = llm.invoke({"texto": usuario_input})
  return result["text"]
  
if __name__ == "__main__":

  criar_base = os.getenv("CRIAR_BASE_VETORIAL", "N")
  if criar_base == "S":
    confirma = input("Confirma criação da base vetorual? (S/N): ")
    if confirma == "S":
      cria_banco_confluence()
    else:
      print("Criação da Base vetorial foi cancelada.")
  else:
    # senão coloca o serviço no ar
    print()
    print("Iniciando servidor...")
    print()
    print("Acesse http://localhost:8108/docs para Swagger")
    print()
    print("Para testar, use o comando abaixo")
    print('curl -X "POST" "http://localhost:8108/obtem_comando_menu/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"texto_usuario\": \"novo produto\"}"')
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=8107)

# para testar
# curl -X "POST" "http://localhost:8106/obtem_comando_menu/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"texto_usuario\": \"novo produto\"}"



