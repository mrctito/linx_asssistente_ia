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


"""
client = QdrantClient(os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API'))
vector_store = Qdrant(client=client, collection_name="openpilot-data", embedding_function)
"""

async def save_vectorstore_qdrant_incremental(chunks: list):
    try:
        nome_col = os.getenv("NOME_BASE_VETORIAL_V2")

        print(f"Salvando base de conhecimento no banco vetorial: {nome_col}")
        print(f'Número total de pacotes a serem gravados: {len(chunks)}')

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        qdrant_client = QdrantClient(
            os.getenv("QDRANT_URL"), 
            prefer_grpc=True,
            timeout=30.0,
            api_key=os.getenv("QDRANT_API_KEY")
        )

        print(f'Preparando para atualizar: {nome_col}')        
        vectorstore = Qdrant(client=qdrant_client,
                             collection_name=nome_col,
                             embeddings=embeddings)
        
        record_manager = SQLRecordManager(f"qdrant/{nome_col}", db_url="sqlite:///record_manager_cache.db")
        record_manager.create_schema()
        
        print(f'Atualizando vetor: {nome_col}')        
        indexing_stats = index(
            chunks,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )

        print("indexing_stats:", indexing_stats)
        print(f"Base de conhecimento foi gravada no banco vetorial (V2): {nome_col}")
    except Exception as e:
        print(f"ERRO ao salvar base de conhecimento no banco vetorial (V2):\n"+str(e)+"\n")    


async def save_vectorstore_qdrant(chunks: list):
    try:
        nome_col = os.getenv("NOME_BASE_VETORIAL_V1")

        print(f"Salvando base de conhecimento no banco vetorial:"+nome_col)
        print(f'Número total de pacotes a serem gravados: {len(chunks)}')
        print(f'Apagando vetor: {nome_col}')        

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        qdrant_client = QdrantClient(
            os.getenv("QDRANT_URL"), 
            prefer_grpc=True,
            timeout=30.0,
            api_key=os.getenv("QDRANT_API_KEY")
        )

        qdrant_client.delete_collection(collection_name=nome_col)
        
        print(f'Recriando vetor: {nome_col}')        
        qdrant_client.recreate_collection(
            collection_name=nome_col,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0,),
            shard_number=2,
        )

        qdrant_doc_store = Qdrant(
            client=qdrant_client, 
            collection_name=nome_col, 
            embeddings=embeddings,
        )

        ultimo_chunk = 0
        while ultimo_chunk < len(chunks):
            chunks_pack = chunks[ultimo_chunk:ultimo_chunk + 500]
            retries = 0

            while retries < 3:
                try:
                    qdrant_doc_store.add_documents(chunks_pack)
                    ultimo_chunk += len(chunks_pack)
                    break
                except Exception as e:
                    if retries == 0: 
                        print("Erro ao salvar chunk: "+str(e))
                    retries += 1
                    if retries <= 3:
                        continue
                    else:
                        raise Exception("Abortando após 5 tentativas.")

        #liga indexação
        print(f'Finalizando atualização do vetor: {nome_col}')
        qdrant_client.update_collection(
            collection_name=nome_col,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

        print(f"Base de conhecimento foi gravada no banco vetorial: {nome_col}")
        print(f'Número total de pacotes que foram gravados: {ultimo_chunk}')
    except Exception as e:
        print(f"ERRO ao salvar base de conhecimento no banco vetorial:\n"+str(e)+"\n")    


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


async def processa_paginas_raiz(id_paginas_raiz: str) -> List[Document]:
    documents = []
    total_palavras = 0  
    total_paginas = 0      

    paginas_raiz = id_paginas_raiz.split(",")
    for pagina_id_raiz in paginas_raiz:
        print(f"Conteúdo da página: {pagina_id_raiz}")
        docs, paginas = await processa_pagina_raiz(pagina_id_raiz)
        total_paginas = total_paginas + paginas
        for doc in docs:
            documents.append(doc)
            palavras = doc.page_content.split()
            total_palavras = total_palavras + len(palavras)
        
    return documents, total_palavras, total_paginas


async def cria_banco_confluence():
    id_paginas_raiz = os.getenv("ID_PAGINAS_RAIZ")
    documents, total_palavras, total_paginas = await processa_paginas_raiz(id_paginas_raiz)

    chunk_size = int(os.getenv("CHUNK_SIZE", "valor_padrao_inteiro"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "valor_padrao_inteiro"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators= ["\n\n", "\n", ".", ";", ",", " ", ""],
                                                length_function=len)
    
    print(f"Foram processadas {total_palavras} palavras em {total_paginas} páginas.")
    chunks = text_splitter.split_documents(documents)
    if chunks is not None:
        await save_vectorstore_qdrant(chunks)
        await save_vectorstore_qdrant_incremental(chunks)
        print(f"Foram processados {len(chunks)} chunks.")
    else:
        print("Nenhum chunk foi processado.")
