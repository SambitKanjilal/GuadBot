import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from scraper import get_part_info

load_dotenv()

llm = init_chat_model("ollama:llama3.2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OllamaEmbeddings(model="llama3.2"),
)
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    parts: list[str]
    rag_context: str
    neo4j_context: str
    api_context: str


class ExtractedParts(BaseModel):
    parts: list[str] = Field(description="Part names or IDs mentioned in the user query")


def extract_parts(state: AgentState):
    last_msg = state["messages"][-1].content
    result = llm.with_structured_output(ExtractedParts).invoke(
        f"Extract all part names or part IDs from this query: {last_msg}"
    )
    return {"parts": result.parts}


def retrieve_rag(state: AgentState):
    query = " ".join(state["parts"]) if state["parts"] else state["messages"][-1].content
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in docs)
    return {"rag_context": context}


def retrieve_neo4j(state: AgentState):
    parts = state["parts"]
    if not parts:
        return {"neo4j_context": ""}

    with neo4j_driver.session() as session:
        if len(parts) == 1:
            # Single part: return it and its immediate neighbours
            rows = session.run(
                "MATCH (p:Part)-[:CONTAINS]-(n:Part) WHERE p.name IN $parts RETURN p.name AS parent, n.name AS child",
                parts=parts,
            )
        else:
            # Multiple parts: find the minimal subgraph that connects all of them.
            # shortestPath finds the path between each pair; the union of all those
            # paths is the smallest set of nodes/edges needed to join every part.
            rows = session.run(
                """
                MATCH path = shortestPath((a:Part)-[:CONTAINS*]-(b:Part))
                WHERE a.name IN $parts AND b.name IN $parts AND id(a) < id(b)
                UNWIND nodes(path) AS n
                WITH collect(DISTINCT n) AS spanning
                UNWIND spanning AS n1
                MATCH (n1)-[:CONTAINS]->(n2)
                WHERE n2 IN spanning
                RETURN n1.name AS parent, n2.name AS child
                """,
                parts=parts,
            )

        connections = [(r["parent"], r["child"]) for r in rows]

    if not connections:
        return {"neo4j_context": f"No assembly connections found for: {', '.join(parts)}"}

    lines = [f"{p} → {c}" for p, c in connections]
    return {"neo4j_context": "Assembly connections:\n" + "\n".join(lines)}


def retrieve_api(state: AgentState):
    parts = state["parts"]
    if not parts:
        return {"api_context": ""}
    results = [f"[{p}]\n{get_part_info(p)}" for p in parts]
    return {"api_context": "\n\n".join(results)}


def synthesize(state: AgentState):
    system = (
        "You are a helpful assistant. Answer the user's question using the context below.\n\n"
        f"## Internal Documentation\n{state['rag_context']}\n\n"
        f"## Assembly Graph\n{state['neo4j_context']}\n\n"
        f"## Distributor Listings\n{state['api_context']}"
    )
    response = llm.invoke([{"role": "system", "content": system}] + state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)

graph_builder.add_node(extract_parts)
graph_builder.add_node(retrieve_rag)
graph_builder.add_node(retrieve_neo4j)
graph_builder.add_node(retrieve_api)
graph_builder.add_node(synthesize)

graph_builder.add_edge(START, "extract_parts")
graph_builder.add_edge("extract_parts", "retrieve_rag")
graph_builder.add_edge("extract_parts", "retrieve_neo4j")
graph_builder.add_edge("extract_parts", "retrieve_api")
graph_builder.add_edge("retrieve_rag", "synthesize")
graph_builder.add_edge("retrieve_neo4j", "synthesize")
graph_builder.add_edge("retrieve_api", "synthesize")
graph_builder.add_edge("synthesize", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": uuid.uuid4()}}

while True:
    user_query = input("Question (or 'quit'): ")
    if user_query.lower() == "quit":
        break
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config=config,
    )
    print("\n" + result["messages"][-1].content + "\n")
