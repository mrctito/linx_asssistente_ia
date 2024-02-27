import os
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import json
from langchain.agents import AgentType
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter,
                                    RecursiveCharacterTextSplitter)
from atlassian import Confluence
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.sitemap import SitemapLoader
from qdrant_client import QdrantClient, models
from langchain.indexes import SQLRecordManager, index

from lib_utils import carregar_arquivo_para_dicionario


async def get_page_content(confluence, space, page_id, page_title, page_url) -> Document:
    try:
        page_pdf = confluence.export_page(page_id)
        buffer = BytesIO(page_pdf)
        pdf_reader = PyPDF2.PdfReader(buffer)
        if pdf_reader:
            num_pages = len(pdf_reader.pages)
            all_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                all_text += text

            buffer.close()

            metadata = {
                "page_id": page_id,
                "space": space,
                "title": page_title, 
                "source": page_url
            }
            return Document(page_content=all_text, metadata=metadata)
        else:
            return Document(page_content="", metadata={})
    except Exception as e:
        print(f"Erro ao tentar obter conteúdo da página {page_id}.\nErro:{str(e)}\n")
        return Document(page_content="", metadata={})


async def processa_pagina_raiz(page_id: str) -> Tuple[List[Document], int]:

    async def processa_pagina(base_url, space, page_id, docs, processed_pages, nivel_atual: int, nivel_maximo: int):
        try:
            if page_id in processed_pages:
                print(f"Evitando processar página {page_id} para evitar loop.")
                return
                    
            page = confluence.get_page_by_id(page_id)
            child_page_id = page.get('id')
            page_title = page.get('title')
            base_url = base_url.rstrip('/')
            webui_link = page['_links']['webui'].lstrip('/') if '_links' in page and 'webui' in page['_links'] else 'URL indisponível'
            page_url = f"{base_url}/{webui_link}"

            #page_url = f"{base_url}{page['_links']['webui']}" if '_links' in page and 'webui' in page['_links'] else 'URL não disponível'
            doc = await get_page_content(confluence, space, child_page_id, page_title, page_url)
            docs.append(doc)
            
            processed_pages.add(page_id)
            print(f"Total parcial de páginas processadas até agora: {len(processed_pages)} - nível atual: {nivel_atual}")

            if nivel_atual >= nivel_maximo:
                return

            all_child_pages = confluence.get_page_child_by_type(page_id, "page", limit=500, start=0)                
            for child_page in all_child_pages:
                await processa_pagina(base_url, space, child_page.get('id'), docs, processed_pages, nivel_atual + 1, nivel_maximo)
        
        except Exception as e:
            print(f"Erro ao processar página {page_id}.\nErro:{str(e)}\n")

    # begin
    confluence = Confluence(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        password=os.getenv("CONFLUENCE_PASSWORD")
    )

    docs = []
    base_url = os.getenv("CONFLUENCE_URL")
    space = confluence.get_page_space(page_id)

    processed_pages = set()
    nivel_maximo = int(os.getenv("CONFLUENCE_NIVEL_RECURSIVIDADE"))
    nivel_atual = 1

    # Processa a página raiz e suas filhas de forma recursiva
    await processa_pagina(base_url, space, page_id, docs, processed_pages, nivel_atual, nivel_maximo)
    print(f"Total de páginas processadas: {len(processed_pages)} para a página raiz {page_id}.")

    return docs, len(processed_pages)


async def processa_paginas_raiz(dados: dict) -> List[Document]:
    documents = []
    total_palavras = 0  
    total_paginas = 0      

    pages_id = dados.get('page_id')
    if pages_id is not None:    
        print("Processando pages_id")
        for pagina_id_raiz in pages_id:
            print(f"Processando página: {pagina_id_raiz}")
            docs, paginas = await processa_pagina_raiz(pagina_id_raiz)
            total_paginas = total_paginas + paginas
            for doc in docs:
                documents.append(doc)
                palavras = doc.page_content.split()
                total_palavras = total_palavras + len(palavras)
    
    print(f"Processadas: {total_paginas} páginas - total {total_palavras} palavras")
    return documents


async def processa_dados_confluence():
    chunk_size = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

    caminho_arquivo = "info_confluence.txt"
    dados = carregar_arquivo_para_dicionario(caminho_arquivo)
    documents = await processa_paginas_raiz(dados)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators= ["\n\n", "\n", ".", ";", ",", " ", ""],
                                                length_function=len)
    
    chunks = text_splitter.split_documents(documents)
    print(f"Confluence: total {len(chunks)} chunks")
    return chunks
