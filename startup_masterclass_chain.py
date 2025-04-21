#-----------------------( for startup masterclass)----------------------------------
# ---------------------------------------------------------------------------
#----------------------( Contextualize question prompt )-----------------------------------
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from helper import get_startup_masterclass_vector_store_retriever, get_llm



def get_startup_masterclass_chain(llm, embedding):

    startup_masterclass_retriever = get_startup_masterclass_vector_store_retriever(embedding)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, startup_masterclass_retriever, contextualize_q_prompt
    )



    #---------------------------( Answer question prompt )---------------------------------------
    # This system prompt helps the AI understand that it should provide comprehensive answers
    # based on the retrieved startup course content
    qa_system_prompt = (
        "You are an expert startup advisor with deep knowledge of the 'How to Start a Startup' course. "
        "Use the following retrieved course content to provide detailed, actionable advice. "
        "Cite specific lectures or concepts from the course when relevant. "
        "If you don't know the answer based on the provided context, acknowledge that limitation. "
        "Focus on practical insights that entrepreneurs can apply immediately. "
        "\n\n"
        "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # ----------------------------( Combining the both above chains )---------------------------------------
    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    startup_masterclass_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return startup_masterclass_rag_chain