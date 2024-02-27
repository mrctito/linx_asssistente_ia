import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import json


def carregar_arquivo_para_dicionario(caminho_arquivo):
    dados = {}
    chave_atual = None
    with open(caminho_arquivo, 'r') as arquivo:
        for linha in arquivo:
            linha = linha.strip()  # Remove espaços em branco e quebras de linha
            if linha.startswith('[') and linha.endswith(']'):
                chave_atual = linha[1:-1]  # Remove os colchetes para usar como chave
                dados[chave_atual] = []  # Inicia uma nova lista para essa chave
            elif chave_atual:
                dados[chave_atual].append(linha)  # Adiciona a linha à lista da chave atual
    return dados
