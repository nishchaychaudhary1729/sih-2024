from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
Don't go beyond the informations given in the data folder.
If the information is not present in the data, just say that you don't know, never make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the model
def load_llm():
    try:
        # Load the locally downloaded model here
        llm = CTransformers(
            model="C:\\Users\\sachi\\OneDrive\\Desktop\\nano\\llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5,
            device_map = 'auto'
        )
        return llm
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None

# QA Model Function
def qa_bot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        if llm is None:
            raise ValueError("LLM loading failed.")
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa
    except Exception as e:
        print(f"Error in QA bot setup: {str(e)}")
        return None


# Output function
def final_result(query):
    qa_result = qa_bot()
    if qa_result:
        response = qa_result({'query': query})
        return response
    else:
        return "Error: QA bot is not initialized properly."

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    if chain:
        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to FAQ Bot. What is your query?"
        await msg.update()

        cl.user_session.set("chain", chain)
    else:
        await cl.Message(content="Error: Failed to initialize the FAQ bot.").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Error: FAQ bot is not initialized.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    try:
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res.get("result", "No result found.")
        # sources = res.get("source_documents", [])

        # if sources:
        #     answer += f"\nSources: {str(sources)}"
        # else:
        #     answer += "\nNo sources found."

        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"Error occurred: {str(e)}").send()
