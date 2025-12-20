from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate


def main():
    # split the document
    pdf_reader = PyPDFLoader("./Transformers/RAGPaper+(1).pdf")
    documents = pdf_reader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)

    # generate embeddings
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)

    # initialize the model
    model = Ollama(model="tinyllama")

    # create chain
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """ Given the following conversation and a follow up question, rephrase 
        the follow up question to be a standalone question.
        Chat History: {chat_history}
        Follow Up Input: {question}
        Standalone question: """
    )

    qa = ConversationalRetrievalChain.from_llm(llm=model, retriever=db.as_retriever(), condense_question_prompt=CONDENSE_QUESTION_PROMPT, verbose=False)

    chat_history = []
    query = "What is a RAG-sequence model?"
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

    
if __name__ == "__main__":
    main()