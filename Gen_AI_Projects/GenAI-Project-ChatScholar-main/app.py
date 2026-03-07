import os, re
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

# --- UPDATED IMPORTS ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Local Embeddings
from langchain_community.llms import Ollama                 # Local Ollama LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Note: We no longer need the 'openai' or 'langchain_openai' imports

# --- CONFIGURATION ---
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

app = Flask(__name__)

# Global variables (initialized as None)
vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

# --- UPDATED LOGIC FUNCTIONS ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    # SWAP: OpenAIEmbeddings -> HuggingFaceEmbeddings
    # "all-MiniLM-L6-v2" is a fast, lightweight local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # SWAP: ChatOpenAI -> ChatOllama
    llm = Ollama(model="deepseek-r1", temperature=0.4)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def _grade_essay(essay):
    # Using Ollama directly for grading function
    llm = Ollama(model="deepseek-r1", temperature=0.4)
    
    system_prompt = f"You are a Chinese bot, grade the essay based on this rubric: {rubric_text}. Respond in Chinese only."
    full_prompt = f"{system_prompt}\n\nESSAY: {essay}"
    
    # Simple invocation for grading
    response = llm.invoke(full_prompt)
    data = response.content
    
    # Convert newlines for HTML display
    return re.sub(r'\n', '<br>', data)


@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history
    msgs = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    if request.method == 'POST':
        if request.form.get('essay_rubric', False):
            global rubric_text
            rubric_text = request.form.get('essay_rubric')

            return render_template('new_essay_grading.html')
        
        if len(request.files['file'].filename) > 0:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            result = _grade_essay(text)
        else:
            text = request.form.get('essay_text')
            result = _grade_essay(text)
    
    return render_template('new_essay_grading.html', result=result, input_text=text)
    
@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    app.run(debug=True)
