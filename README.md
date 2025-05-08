# ReAct Chain AI Assistant

This project implements an intelligent assistant that leverages LangChain's ReAct framework to provide accurate information from multiple specialized knowledge bases.

## Overview

The assistant is designed to answer questions by reasoning through which knowledge source is most appropriate and retrieving relevant information. It employs a retrieval-augmented generation (RAG) approach with three distinct knowledge domains:

1. **Startup Data** - Company and business information about startups
2. **Legal Data** - Legal information, regulations, and documents
3. **StartupMasterclass** - Comprehensive guidance from the "How to Start a Startup" course materials

## Architecture

The system is built on the following components:

- **LangChain's ReAct Framework** - Enables the AI to reason through complex questions and select appropriate tools
- **Gemini 1.5 Flash** - Powers natural language understanding and generation
- **HuggingFace Embeddings** - Creates vector representations of text using the "sentence-transformers/all-mpnet-base-v2" model
- **Chroma Vector Database** - Stores and retrieves vectorized documents efficiently
- **Streamlit UI** - Provides a user-friendly interface for interacting with the assistant

## Project Structure

```
ReAct_Chain_AI_Assistant/
├── startup_chain.py        # Chain for startup company data
├── legal_chain.py          # Chain for legal data
├── startup_masterclass_chain.py  # Chain for startup course content
├── helper.py               # Utility functions for the models and retrievers
├── st_app.py                 # Main application script with the ReAct agent setup
├── process_documents.py    # Script to process documents and create vector stores
├── VectorStores/           # Vector database storage
│   ├── startup_data_vector_store/
│   ├── legal_data_vector_store/
│   └── startup_masterclass_vector_store/
└── Data/                   # Source data
    ├── cleaned/            # Processed text files
    └── startup_course.pdf  # PDF of "How to Start a Startup" course
```

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AvivaShooman/GenAI-Assignment-7.git
   cd GenAI-Assignment-7
   ```

2. **Create and activate a virtual environment**
   ```bash
   conda create -n bot python=3.10
   conda activate bot
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Process documents to create vector stores**
   ```bash
   python process_documents.py
   ```

6. **Run the application**
   ```bash
   python -m streamlit run st_app.py
   ```

## Usage

1. Open the Streamlit app in your browser (typically at http://localhost:8501)
2. Type your question in the chat input
3. The assistant will determine which knowledge base is most relevant and provide an answer
4. You can ask follow-up questions as the system maintains conversation history

## Sample Questions

### Startup Data
- "What are the most funded AI startups?"
- "When was Tesla founded and who is the CEO?"

### Legal Data
- "What are the key considerations in drafting a privacy policy?"
- "What regulations apply to SaaS companies collecting user data?"

### StartupMasterclass
- "What does the startup course teach about identifying good startup ideas?"
- "How should I approach fundraising according to the startup masterclass?"
- "What are the counterintuitive parts of startups mentioned in the course?"

## Extending the System

To add additional knowledge domains:
1. Create a new vector store with your documents
2. Implement a new chain in a separate file
3. Add a retriever function in helper.py
4. Register the new tool in st_app.py
5. Update the UI in st_app.py to reflect the new capability

## License

[MIT License]

## Acknowledgments

- The "How to Start a Startup" course materials
- LangChain for the ReAct framework and retrieval tools
- Google for the Gemini model
