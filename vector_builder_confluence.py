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


def save_vectorstore_qdrant(chunks: list):
    try:
        print(f"Salvando base de conhecimento no banco vetorial:"+os.getev("NOME_BASE_VETORIAL"))
        print(f'Número total de pacotes a serem gravados: {len(chunks)}')

        embeddings = OpenAIEmbeddings(openai_api_key=base_conhecimento.openai_api_key)

        qdrant_client = QdrantClient(
            base_conhecimento.qadrant_url, 
            prefer_grpc=True,
            timeout=30.0,
            api_key=base_conhecimento.qadrant_api_key
        )

        nome_col = get_collection_name(base_conhecimento)
        logger.log(f'Apagando vetor: {nome_col}')        
        qdrant_client.delete_collection(collection_name=nome_col)
        
        logger.log(f'Recriando vetor: {nome_col}')        
        qdrant_client.recreate_collection(
            collection_name=nome_col,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0,),
            shard_number=2,
        )

        qdrant_doc_store = Qdrant(
            client=qdrant_client, 
            collection_name=get_collection_name(base_conhecimento), 
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
                        logger.log("Erro ao salvar chunk: "+str(e))
                    retries += 1
                    if retries <= 3:
                        continue
                    else:
                        raise Exception("Abortando após 5 tentativas.")

        #liga indexação
        logger.log(f'Finalizando atualização do vetor')        
        qdrant_client.update_collection(
            collection_name=get_collection_name(base_conhecimento),
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

        logger.log(f"Base de conhecimento foi gravada no banco vetorial.")
        logger.log(f'Número total de pacotes que foram gravados: {ultimo_chunk}')
    except Exception as e:
        raise Exception(f"Erro ao salvar base de conhecimento no banco vetorial:\n"+str(e)+"\n")    


def get_page_content(confluence, space, page_id, page_title, page_url, logger) -> Document:
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
        logger.log(f"Erro ao tentar obter conteúdo da página {page_id}.\nErro:{str(e)}\n")
        return Document(page_content="", metadata={})


def processa_pagina_raiz(page_id: str) -> Tuple[List[Document], int]:

    def processa_pagina(base_url, space, page_id, docs, processed_pages, nivel_atual, nivel_maximo):
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
            doc = get_page_content(confluence, space, child_page_id, page_title, page_url)
            docs.append(doc)
            
            processed_pages.add(page_id)
            print(f"Total parcial de páginas processadas até agora: {len(processed_pages)}", True)

            if nivel_atual >= nivel_maximo:
                return

            all_child_pages = confluence.get_page_child_by_type(page_id, "page", limit=500, start=0)                
            for child_page in all_child_pages:
                processa_pagina(base_url, space, child_page.get('id'), docs, processed_pages, nivel_atual + 1, nivel_maximo)
        
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
    nivel_maximo = os.getenv("CONFLUENCE_NIVEL_RECURSIVIDADE")
    nivel_atual = 1

    # Processa a página raiz e suas filhas de forma recursiva
    processa_pagina(base_url, space, page_id, docs, processed_pages, nivel_atual, nivel_maximo)
    print(f"Total de páginas processadas: {len(processed_pages)} para a página raiz {page_id}.")

    return docs, len(processed_pages)


def processa_paginas_raiz(id_paginas_raiz: str) -> List[Document]:
    documents = []
    total_palavras = 0        

    paginas_raiz = id_paginas_raiz.split(",")
    for pagina_id_raiz in paginas_raiz:
        print(f"Conteúdo da página: {pagina_id_raiz}")
        docs, total_paginas = processa_pagina_raiz(pagina_id_raiz)
        for doc in docs:
            documents.append(doc)
            palavras = doc.page_content.split()
            total_palavras = total_palavras + len(palavras)
        
    return documents, total_palavras, total_paginas


def cria_banco_confluence():
    chunks_array = []
    id_paginas_raiz = os.getenv("ID_PAGINAS_RAIZ")
    documents, total_palavras, total_paginas = processa_paginas_raiz(id_paginas_raiz)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, 
                                                chunk_overlap=75,
                                                separators= ["\n\n", "\n", ".", ";", ",", " ", ""],
                                                length_function=len)
    
    print(f"Foram processadas {total_palavras} palavras em {total_paginas} páginas.")
    chunks = text_splitter.split_documents(documents)
    if chunks is not None:
        save_vectorstore_qdrant(chunks)
        print(f"Foram processados {len(chunks)} chunks.")
    else:
        print("Nenhum chunk foi processado.")
