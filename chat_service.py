import os
import json
from pydantic import BaseModel
from typing import Tuple, List
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from openai import Embedding
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import (ConversationBufferWindowMemory, SQLChatMessageHistory)
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

condense_template='''
"""Given the following chat history and a follow up question, rephrase the follow up input question 
to be a standalone question. Or end the conversation if it seems like it's done.

Conversation History:
{chat_history}

Follow Up Input:
{question}

Standalone question:"""
'''

prompt_template_restrito_ao_contexto = """
\nPRESTE MUITA ATENÇÃO E SIGA TODAS ESSAS INSTRUÇÕES CUIDADOSAMENTE:
1- Baseie suas respostas exclusivamente no conteúdo fornecido no CONTEXTO apresentado.
2- Forneça respostas completas e com todos os detalhes e informações complementares que estiverem disponíveis no contexto.
3- Não procure ou inclua informações além das fornecidas no contexto.
4- Evite especulações; forneça respostas fundamentadas somente nas informações disponíveis.
5- Nunca apresente informações inventadas ou fictícias.
6- Responda sempre no idioma em que a pergunta foi feita. Caso o contexto esteja em um idioma diferente, busque esclarecer o significado mantendo a integridade da informação.
7- Se não for possível fornecer uma resposta devido à falta de informações ou ambiguidade no contexto, responda com: "Desculpe, não tenho informações suficientes para responder a essa pergunta com precisão."

Lembre-se de que em situações onde o contexto possa ser interpretado de várias maneiras, considere mencionar essa ambiguidade em sua resposta para manter a transparência.
"""

prompt_template_nao_restrito_ao_contexto = """
\nSIGA ESTAS INSTRUÇÕES COM ATENÇÃO PARA FORMULAR SUA RESPOSTA:
1- Baseie suas respostas no CONTEXTO fornecido, complementando com seu conhecimento quando apropriado.
2- Use informações adicionais do seu conhecimento apenas quando estas enriquecerem a resposta e estiverem em harmonia com o contexto apresentado.
3- Forneça respostas completas e com todos os detalhes e informações complementares que estiverem disponíveis.
4- Seja claro sobre qual parte da resposta vem do contexto e qual parte é baseada em conhecimento externo.
5- Evite especulações e não apresente informações fictícias.
6- Responda no idioma em que a pergunta foi feita. Caso o contexto esteja em um idioma diferente, faça a tradução necessária mantendo a precisão da informação.
7- Se, mesmo com a inclusão de seu conhecimento, a informação necessária não estiver disponível ou o contexto for ambíguo, responda com: "Desculpe, não tenho informações suficientes para responder a essa pergunta com total precisão."

Lembre-se de utilizar seu conhecimento de forma responsável e apenas quando isso contribuir para a precisão e relevância da resposta. Sua transparência é essencial para manter a confiança na qualidade das informações fornecidas.
"""

human_prompt_template='''
Pergunta:
{question}

Resposta:
'''


def get_base_prompt_template() -> str:
    base_prompt_template = """
    Você é um assistente útil, atencioso e muito educado. Especialista no sistema Linx Seller Web. 
    Sua missão é responder as perguntas dos clientes de forma didática, com informações precisas e relevantes.
    """

    #controla liberdade de acesso ao próprio conhecimeto do GPT
    if os.getenv("RESTRITO_AO_CONTEXTO", "S") == "S":
        base_prompt_template += prompt_template_restrito_ao_contexto
    else:
        base_prompt_template += prompt_template_nao_restrito_ao_contexto
            
    return base_prompt_template 


async def GetPrompts() -> Tuple[ChatPromptTemplate, PromptTemplate]:
    try:
        base_prompt_template = get_base_prompt_template()
        template = base_prompt_template + "\n\nHISTÓRICO DA CONVERSA:{chat_history}.\nFIM DO HISTÓRICO DA CONVERSA.\n\nINFORMAÇÕES DE CONTEXTO:{context}.\nFIM DO CONTEXTO."
        prompt_template = PromptTemplate(template=template, input_variables=["chat_history", "context"])
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt_template)

        template = human_prompt_template
        prompt_template = PromptTemplate(template=template, input_variables=["question"])
        user_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
        condense_prompt = PromptTemplate(template=condense_template, input_variables=["chat_history", "question"])

        return chat_prompt, condense_prompt
    except Exception as e:
        print("Erro ao criar prompts: "+str(e))
        raise Exception("Erro ao criar prompts: "+str(e))


async def GetChatModel(streaming: bool=False) -> ChatOpenAI:
    llm = ChatOpenAI(temperature=0, 
                    model=os.getenv("MODEL_NAME"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    streaming=streaming,
                    verbose=(os.getenv("VERBOSE", "S") == "S")
                    )
    return llm


async def GetRetriever() -> VectorStoreRetriever:
    try:
        collection_name = os.getenv("NOME_BASE_VETORIAL_ATIVA")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        qdrant_client = QdrantClient(os.getenv("QDRANT_URL"), 
                                        api_key=os.getenv("QDRANT_API_KEY"), 
                                        prefer_grpc=True
                                        )
        
        vectorStore = Qdrant(client=qdrant_client,
                            collection_name=collection_name,
                            embeddings=embeddings
                            )
        
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

        return retriever
    except Exception as e:
        print("Erro GetRetriever:", str(e))
        raise Exception("Erro GetRetriever: "+str(e))


async def GetMemory() -> BaseChatMessageHistory:
    llm = await GetChatModel()

    connection = os.getenv("DB_CACHE_URL")
    table_name = "linx_seller_chat_history"
    cache_id = os.getenv("EMAIL_USUARIO")

    chat_history = SQLChatMessageHistory(session_id=cache_id, 
                                         connection_string=connection, 
                                         table_name=table_name
                                         )
    
    memory_conversation = ConversationBufferWindowMemory(llm=llm, 
                                            max_token_limit=1000,
                                            output_key='answer',
                                            memory_key="chat_history",
                                            chat_memory=chat_history,
                                            return_messages=True
                                            )
    """
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    """

    return memory_conversation


async def GetConversationChain() -> ConversationalRetrievalChain:
    memory = await GetMemory()
    retriever = await GetRetriever()
    chat_prompt, condense_prompt = await GetPrompts()

    llm_question_generator = await GetChatModel()     
    question_generator_chain = LLMChain(llm=llm_question_generator,
                                        prompt=condense_prompt,
                                        verbose=(os.getenv("VERBOSE", "S") == "S")
                                        )

    llm = await GetChatModel()
    combine_docs_chain = load_qa_chain(llm=llm,
                                     chain_type="stuff",
                                     prompt=chat_prompt,
                                     verbose=(os.getenv("VERBOSE", "S") == "S")
                                    )

    chain = ConversationalRetrievalChain(retriever=retriever, 
                                        memory=memory,
                                        rephrase_question=False,
                                        return_generated_question=False,
                                        return_source_documents=True,
                                        combine_docs_chain=combine_docs_chain,
                                        question_generator=question_generator_chain,
                                        verbose=(os.getenv("VERBOSE", "S") == "S")
                                        )
    
    return chain


async def GetConversationChainRunnable():

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    chat_prompt, condense_prompt = await GetPrompts()    
    model1 = await GetChatModel()
    model2 = await GetChatModel()
    retriever = await GetRetriever()


    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT | model1 | StrOutputParser(),
    }
            
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | model2 | StrOutputParser(),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain


def concatena_sources(documents: List[Document]) -> str:
    # Usa compreensão de lista para extrair o 'source' de cada document.metadata
    sources = [doc.metadata['source'] for doc in documents if 'source' in doc.metadata]
    # Concatena todos os sources com '\n' como separador
    return "\n".join(sources)


def monta_resposta_chat(response) -> str:
    all_sources = concatena_sources(response["source_documents"])
    answer = response["answer"]
    return answer + "\n\n" + all_sources


async def chat(query):
    chain = await GetConversationChain()
    response = await chain.ainvoke({"question": query})
    result = monta_resposta_chat(response)
    return result
