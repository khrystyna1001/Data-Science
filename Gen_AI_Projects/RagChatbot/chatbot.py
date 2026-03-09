from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

def main():
    llm = OllamaLLM(model="llama3.1")

    response = llm.invoke("Hello")
    print(response)

    pdf_reader = PyPDFLoader("Gen_AI_Projects/RagChatbot/RAGPaper.pdf")
    documents = pdf_reader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow up Input: {question}
    Standalone questions: """)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False
    )

    chat_history=[]
    query="""What is RAGs and tell me more about use cases of RAGs, in a detailed manner"""
    result = qa({"question":query,"chat_history":chat_history})
    print(result["answer"])

if __name__ == "__main__":
    main()
