from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from pinecone_notebooks.colab import Authenticate
from langchain_pinecone import PineconeVectorStore

def main():
    OPENAI_API_KEY = "KEY"
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    llm.invoke("What is EDA?")

    loader = PyPDFLoader("RAGPaper.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, )
    chunks = text_splitter.split_documents(pages)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    embedded_query = embeddings.embed_query("What is full form of RAG?")

    print(f"Embedding length: {len(embedded_query)}")
    print(embedded_query[:10])

    sentence1 = embeddings.embed_query("I love to watch youtube videos")
    sentence2 = embeddings.embed_query("Subscribe to My YouTube Channel")

    query_sentence1_similarity = cosine_similarity([embedded_query], [sentence1])[0][0]
    query_sentence2_similarity = cosine_similarity([embedded_query], [sentence2])[0][0]

    # vector db
    pc = Pinecone(api_key="PINECONE_API_KEY")

    cloud = 'aws'
    region = 'us-east-1'

    spec = ServerlessSpec(cloud=cloud, region=region)

    index_name = 'ragdemo'
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=1536, 
            metric='cosine',
            spec=spec
        )

    # connect to index
    index = pc.Index(index_name)
    Authenticate()

    docsearch = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

    qa_pinecone = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever=docsearch.as_retriever(),
                                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                           return_source_documents=True,
                                           verbose=False)

    chat_history = []
    query = """?What is RAG?"""
    result = qa_pinecone({"question": query, "chat_history": chat_history})
    print(result["answer"])

if __name__ == "__main__":
    main()