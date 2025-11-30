from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    llm.invoke("Explain EDA")

    pdf_reader = PyPDFLoader("./RAGPaper+(1).pdf")
    documents = pdf_reader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)

    # create chain
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """Given the following conversation and a follow up question, 
        rephrase the follow up question
        to be a standalone question.
        
        Chat History:
        {chat_history}
        
        Follow Up Input:
        {question}
        
        Standalone questions:"""
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False,
    )

    chat_history = []
    query = """Does the vendor have experience with similar industries and use cases?"""
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

    chat_history = []
    query = """Does the vendor's financial offer make sense compared to the timeline, resources, and deliverables?"""
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

    chat_history = []
    query = """What is RAGs?"""
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

if __name__ == "__main__":
    main()