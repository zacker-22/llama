from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
import os
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("MY_DB_URL")
def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else "groq_rag_documents_openai"
    )

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 2 references are added to the prompt
            num_documents=10,
        ),
        description="You are an AI called SFBUBot. You are here to help answer questions about the SFBU university.",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Answer the question from context only.",
            "Do not hallucinate or make up information. If you do not know the answer, say 'I don't know.'",
            "Ask for clarification if you need more information to answer the question.",
        ],
        # This setting adds references from the knowledge_base to the user prompt
        add_references_to_prompt=False,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 4 previous messages from chat history to the messages
        num_history_messages=100,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )