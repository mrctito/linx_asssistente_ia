import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


PROMPT_BASE = '''
Você é um assistente inteligente que analisa comandos relacionados a operações comerciais.
Sua tarefa é decompor cada comando em suas partes constituintes, identificando a ação, produto, 
fornecedor, cliente, serviço, data, quantidade, preço, unidade de medida, motivo, endereço,  
número da nota fiscal, número do pedido, e quaisquer outros elementos relevantes que façam sentido 
para o pedido.

Comando: {texto}
Resposta:
'''

def prepara_prompt():
    prompt = PromptTemplate(
        template=PROMPT_BASE,
        input_variables=["texto"],
    )
    return prompt
