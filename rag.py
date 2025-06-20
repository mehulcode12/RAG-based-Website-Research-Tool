from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

# Constants
CHUNK_SIZE = 800  
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"  
COLLECTION_NAME = "rag" 
llm = None
vector_store = None

# Initialize components
def initialize_components():

    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=2000
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english"),
            persist_directory=str(Path("vector_db"))
        )


def process_urls(urls):
    """
    scraps the data from a url and stores them in a vector db
    """

    yield "███ Processing URLs:"
    yield "████ Initializing components..."
    initialize_components()

    yield "█████ Components initialized."

    vector_store.reset_collection()

    yield "███████ loader initialized."
    #loader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    yield "█████████ text splitter initialized."
    #text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", "."],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=40
    ) 
    docs = text_splitter.split_documents(data)

    yield "████████████ vector store initialized."
    vector_store.add_documents(docs, ids=[str(uuid4()) for _ in range(len(docs))])

def generate_answers(query):
    """
    Generates answers based on the query using the vector store.
    """
    if not vector_store:
        raise RuntimeError("!*!*!*!*!*!*!*!*!*!*!*Vector store is not initialized")
    print("██████████████ Generating answers for query:", query)
    initialize_components()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"question": query})
    return result['answer'], result.get("sources", [])


if __name__ == "__main__":
    urls = [
        "https://www.cnbctv18.com/personal-finance/rbi-mpc-repo-rate-cut-50-bps-home-loan-interest-emis-cheaper-buying-purchase-june-19616651.htm",
        "https://medium.com/@mehulligade12/who-am-i-and-why-i-write-about-machine-learning-and-ai-m001-mehul-ligade-c3695555ddfd"
    ]

    process_urls(urls)

    answer, source = generate_answers("What did chariman of Womeki Group said?")
    print("Answer:", answer)
    print("Sources:", source)