import os
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import Runnable
import chainlit as cl
from chat_service import GetConversationChain, GetConversationChainRunnable, monta_resposta_chat
from langchain.callbacks.base import BaseCallbackHandler

###chainlit run chat_interface.py


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content=f"Olá, o que você deseja saber?", disable_feedback=True)
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()
    runnable = await GetConversationChain(True, cl)
    res = await runnable.ainvoke({"question": message.content})
    reposta = monta_resposta_chat(res)
    await cl.Message(content=reposta).send()
