import os
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from openai import Embedding
from langchain.memory import (ConversationBufferWindowMemory, SQLChatMessageHistory)
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate



async def GetPrompt():
    system_template = """
    Você é um assistente útil, atencioso e muito educado. Sua missão é responder as perguntas dos clientes da melhor forma possível.
    Você deve utilizar as informações disponíveis no contexto e na conversa até o momento para responder as perguntas. 
    Você não pode utilizar o seu conhecimento para responder as perguntas. 
    Você não pode acessar a internet para responder as perguntas.
    Você não pode inventar informações.
    Você não pode inventar informações ficticias. 
    Se você não conseguir responder às perguntas com base no contexto e na conversa, você deve apenas dizer que não consegue responcer.

    Context: {context}
    Conversation so far: {chat_history}
    Your question: {question}
    """

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(input_variables=["context", "chat_history", "question"],
                                        template=system_template)),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    return prompt


async def GetRetriever() -> VectorStoreRetriever:
    try:
        collection_name = os.getenv("NOME_BASE_VETORIAL")
        vectorstore = None
        retriever = None
        try:
            collection_name = collection_name
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

            qdrant_client = QdrantClient(os.getenv("QDRANT_URL"), 
                                         api_key=os.getenv("QDRANT_API_KEY"), 
                                         prefer_grpc=True)
            
            vectorStore = Qdrant(client=qdrant_client,
                            collection_name=collection_name,
                            embeddings=embeddings)
            
            SEARCH_K = int(os.getenv("SEARCH_K"))
            SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE"))
            retriever = vectorStore.as_retriever(search_type="similarity", 
                                        search_kwargs={"k":SEARCH_K, "score_threshold": SEARCH_MIN_SCORE})
            
            """
            retriever = ScoreThresholdRetriever.fromVectorStore(
                vectorStore,
                minSimilarityScore=0.9,  # Finds results with at least this similarity score
                maxK=100,  # The maximum K value to use. Use it based to your chunk size to make sure you don't run out of tokens
                kIncrement=2  # How much to increase K by each time. It'll fetch N results, then N + kIncrement, then N + kIncrement * 2, etc.
            )
            """

            print(f"Qdrant Collection:{collection_name}")
        except Exception as e:
            raise Exception("Erro ao criar retriever QDRANT: "+str(e))

        return retriever
    except Exception as e:
        print("Erro GetRetriever:", str(e))
        raise Exception("Erro GetRetriever: "+str(e))


async def GetMemoty() -> BaseChatMessageHistory:
    llm = ChatOpenAI(temperature=0, 
                     model=os.getenv("MODEL_NAME"),
                     openai_api_key=os.getenv("OPENAI_API_KEY"),
                     streaming=False)

    connection = os.getenv("DB_CACHE_URL")
    table_name = "linx_seller_chat_history"
    cache_id = os.getenv("EMAIL_USUARIO")
    chat_history = SQLChatMessageHistory(session_id=cache_id, connection_string=connection, table_name=table_name)
    memory_conversation = ConversationBufferWindowMemory(llm=llm, 
                                            max_token_limit=1000,
                                            output_key='answer',
                                            memory_key="chat_history",
                                            chat_memory=chat_history,
                                            return_messages=True)
    """
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    """

    return memory_conversation


async def GetConversationChain() -> ConversationalRetrievalChain:
    memory = await GetMemoty()
    retriever = await GetRetriever()

    llm = ChatOpenAI(temperature=0, 
                     model=os.getenv("MODEL_NAME"),
                     openai_api_key=os.getenv("OPENAI_API_KEY"),
                     streaming=False)

    prompt = await GetPrompt()

    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  chain_type="stuff",
                                                  retriever=retriever,
                                                  memory=memory,
                                                  return_source_documents=True,
                                                  combine_docs_chain_kwargs={"prompt": prompt})
    
    return chain


async def chat(query):
    chain = await GetConversationChain()
    response = await chain.ainvoke({"question": query})
    print(response)
    return response["answer"]
