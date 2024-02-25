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


async def GetRetriever(embeddings: Embedding) -> VectorStoreRetriever:
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



def chat():
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

