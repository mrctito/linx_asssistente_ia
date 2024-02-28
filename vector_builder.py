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

from vector_builder_confluence import processa_dados_confluence
from vector_builder_youtube import processa_dados_youtube


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
                    print("Gravando chunk: "+str(ultimo_chunk))
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
