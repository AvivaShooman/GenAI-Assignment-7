import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def get_embeddings_model():
    # Explicitly set the device
    device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    print("\n [LOG] OK !! get_embeddings_model DONE \n")
    return embeddings


def get_legal_data_vector_store_retriever(embedding):
    persistent_directory = "./VectorStoresOld/legal_data_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("\n [LOG] OK !! get_legal_data_vector_store_retriever DONE \n")
    return retriever


def get_startup_data_vector_store_retriever(embedding):
    persistent_directory = "./VectorStoresOld/startup_data_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("\n [LOG] OK !! get_startup_data_vector_store_retriever DONE \n")
    return retriever


def get_startup_masterclass_vector_store_retriever(embedding):
    persistent_directory = "./VectorStoresOld/startup_masterclass_vector_store"
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("\n [LOG] OK !! get_startup_masterclass_vector_store_retriever DONE \n")
    return retriever


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    print("\n [LOG] OK !! get_llm DONE \n")
    return llm