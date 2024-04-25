from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



with open(r'C:\Users\dsai9\Projects\RAG_Application\GEMINI_API_KEY.txt','r') as key:
    GOOGLE_API_KEY=key.read().strip()

loader = PyPDFLoader(r"C:\Users\dsai9\Projects\RAG_Application\data\LeaveNoContextBehind.pdf")
data= loader.load_and_split()

# Split the document into chunks

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(data)

# Creating Chunks Embedding
# We are just loading OpenAIEmbeddings



embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, 
                                               model="models/embedding-001")

# Store the chunks in vector store


# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory=r"C:\Users\dsai9\Projects\RAG_Application\ChromaDB")

# Persist the database on drive
db.persist()

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory=r"C:\Users\dsai9\Projects\RAG_Application\ChromaDB", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

print(type(retriever))

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])



chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, 
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def getResponse(question):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )


    response = rag_chain.invoke(question)
    return response

