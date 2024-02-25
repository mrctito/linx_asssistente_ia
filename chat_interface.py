import os
from typing import List
from langchain_core.documents import Document
from langchain.chains import LLMChain, ConversationalRetrievalChain
import chainlit as cl
from chat_service import GetConversationChain, monta_resposta_chat

#chainlit run chat_interface.py


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content=f"Iniciando ...", disable_feedback=True)
    await msg.send()
    chain = await GetConversationChain()
    cl.user_session.set("chain", chain)
    msg = cl.Message(content=f"Pronto.", disable_feedback=True)
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    #res = await chain.ainvoke(message.content, callbacks=[cb])
    res = await chain.ainvoke({"question": message.content}, callbacks=[cb])
    reposta = monta_resposta_chat(res)
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    resposta = monta_resposta_chat(res)

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
