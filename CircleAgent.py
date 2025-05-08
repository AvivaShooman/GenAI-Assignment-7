from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool

# Application modules
from startup_chain import get_startup_data_chain
from legal_chain import get_legal_data_chain
from startup_masterclass_chain import get_startup_masterclass_chain
from helper import get_llm, get_embeddings_model


llm = get_llm()
embedding = get_embeddings_model()

startup_data_rag_chain = get_startup_data_chain(llm, embedding)
legal_data_rag_chain = get_legal_data_chain(llm, embedding)
startup_masterclass_rag_chain = get_startup_masterclass_chain(llm, embedding)  # Initialize the new chain


# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react")


def invoke_startup_data_rag_chain(input, **kwargs):
    return startup_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})


def invoke_legal_data_rag_chain(input, **kwargs):
    return legal_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})


def invoke_startup_masterclass_chain(input, **kwargs):
    return startup_masterclass_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})  # Fixed this line


tools = [
    Tool(
        name="Startup Data Answer Question",
        func=invoke_startup_data_rag_chain,
        description="useful for when you need to answer questions about startups, business data, or company information",
    ),
    Tool(
        name="Legal Data Answer Question",
        func=invoke_legal_data_rag_chain,
        description="useful for when you need to answer questions about legal information, regulations, or legal documents",
    ),
    Tool(
        name="StartupMasterclass",
        func=invoke_startup_masterclass_chain,
        description="useful for when you need expert guidance on startup fundamentals, product development, fundraising, growth tactics, team building, operations, and entrepreneurship strategies from comprehensive course materials",
    )
]



# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))


