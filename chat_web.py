import os
from typing import List
from langchain_core.documents import Document
from langchain.chains import LLMChain, ConversationalRetrievalChain
import chainlit as cl
from chat_service import GetConversationChain, monta_resposta_chat

###chainlit run chat_interface.py

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
    res = await chain.ainvoke({"question": message.content}, callbacks=[cb])
    reposta = monta_resposta_chat(res)
    await cl.Message(content=reposta).send()
