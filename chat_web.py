import os
from typing import List
from langchain_core.documents import Document
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import Runnable
import chainlit as cl
from chat_service import GetConversationChain, GetConversationChainRunnable, monta_resposta_chat

###chainlit run chat_interface.py

@cl.on_chat_start
async def on_chat_start():
    '''
    chain = await GetConversationChain()
    cl.user_session.set("chain", chain)
    '''

    runnable = await GetConversationChainRunnable()
    cl.user_session.set("runnable", runnable)

    msg = cl.Message(content=f"Olá, o que você deseja saber?", disable_feedback=True)
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):

    '''
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.LangchainCallbackHandler() #AsyncLangchainCallbackHandler()
    res = chain.ainvoke({"question": message.content}, callbacks=[cb])
    reposta = monta_resposta_chat(res)
    await cl.Message(content=reposta).send()
    '''

    runnable = cl.user_session.get("runnable")  # type: Runnable

    inputs = {"question": message.content}
    result = await runnable.ainvoke(inputs)
    res = result["answer"]
    reposta = monta_resposta_chat(res)
    msg = cl.Message(content=reposta, disable_feedback=True)
    await msg.send()

    return 

    answer_prefix_tokens=["FINAL", "answer"]
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True,
                                                                     force_stream_final_answer=True,
                                                                     answer_prefix_tokens=answer_prefix_tokens
                                                                     )
                                        ]
                            ),
    ):
        await msg.stream_token(chunk)

    await msg.send()
