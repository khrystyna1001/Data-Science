from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from bert_score import BERTScorer
import torch


def main():
    # Read the file and split the document into chunks
    pdf_reader = PyPDFLoader("./Transformers/Power+BI+Ebook.pdf")
    documents = pdf_reader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Calling the model
    eval_llm = Ollama(model="tinyllama")

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """ Given the following conversation and a follow up question, rephrase 
        the follow up question to be a standalone question.
        Chat History: {chat_history}
        Follow Up Input: {question}
        Standalone question: """
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=eval_llm, 
        retriever=db.as_retriever(), 
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, 
        return_source_documents=True, 
        verbose=False
    )

    # RAG Evaluation
    sample_queries = [
        {
            "question": "What are the main components of Power BI?",
            "answer": "The main components of Power BI are Power BI Desktop, Power BI Service, and Power BI Mobile.",
            "reference": "Power BI has three main components: Desktop (Windows app for creating reports), Service (cloud platform for sharing and collaboration), and Mobile (app for viewing reports on mobile devices)."
        },
        {
            "question": "What are the key features of Power BI Desktop?",
            "answer": "Key features include data connection and transformation, data modeling, interactive visualizations, report publishing, and collaboration tools.",
            "reference": "Power BI Desktop allows users to get data, analyze it with DAX, visualize using 150+ visuals, publish to cloud or on-premises, and collaborate with team members."
        },
        {
            "question": "How does Power BI integrate with Excel?",
            "answer": "Power BI can analyze data in Excel, import Excel data, connect to Excel workbooks, and pin Excel ranges to dashboards.",
            "reference": "Power BI integrates with Excel by enabling analysis in Excel, data import, connection to workbooks, and uploading Excel files for dashboard pinning."
        },
        {
            "question": "What is the Power Query Editor used for?",
            "answer": "Power Query Editor is used for data transformation and cleansing before importing data into models or visualizations.",
            "reference": "Power Query Editor includes tools like ribbon tabs, data preview, query settings, and footer with data stats for transforming and shaping data."
        },
        {
            "question": "How can you group data in Power BI?",
            "answer": "You can group data using the 'Group By' feature in the Transform tab of the Query Editor by selecting columns and specifying aggregate calculations.",
            "reference": "In Power BI's Query Editor, the Group By dialog allows users to select columns and apply aggregate functions to group data."
        }
    ]

    df = pd.DataFrame(sample_queries)

    retriever = db.as_retriever()

    def process_query(query):
        chat_history = []
        result = qa({"question": query, "chat_history": chat_history})
        relevant_docs = retriever.invoke(query)
        print(result['answer'])
        return result['answer'], relevant_docs

    # RAGAS Framework
    results = []
    for _, row in df.iterrows():
        question = row['question']
        ground_truth = row['answer']

        answer, relevant_docs = process_query(question)

        results.append({
            "user_input": question,
            "reference": ground_truth,
            "response": answer,
            "retrieved_contexts": [relevant_docs[0].page_content]
        })

    # Intialize the BART model and tokenizer
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Intialize the BERTScorer
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    bert_scores = []
    bart_scores = []
    
    for item in results:
        inputs = bart_tokenizer(item["response"], truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            bart_score = bart_model(**inputs).logits
        bart_scores.append(bart_score.mean().item())

        P, R, F1 = bert_scorer.score([item["response"]], [item["reference"]])
        bert_scores.append(F1.numpy().mean())

    print("BERT Scores:", bert_scores)
    print("BART Scores:", bart_scores)

if __name__ == "__main__":
    main()