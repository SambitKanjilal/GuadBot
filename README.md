# guad-agent-handmade

Parts information agent. Given a user question, identifies relevant parts, expands them via an assembly graph, pulls context from internal docs, Neo4j, and distributor listings (Mouser/Digikey), then either synthesizes an answer or returns raw data.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.com) with `llama3.2` pulled (`ollama pull llama3.2`)

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd guad_agent_handmade
uv sync
uv run playwright install chromium
```

**2. Start databases**
```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword -d neo4j:latest
docker run --name postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=parts -p 5432:5432 -d postgres
```

**3. Configure environment**
```bash
cp .env.example .env
```
Edit `.env` with your Neo4j password and Postgres DSN.

**4. Ingest data**
```bash
# Drop .txt files into internal_documents/ then:
uv run ingest.py

# Drop .step files into step_files/ then:
uv run ingest_step.py
```

**5. Run**
```bash
uv run main.py
```

## Environment variables

| Variable | Description |
|---|---|
| `NEO4J_URI` | Neo4j bolt URI (default: `bolt://localhost:7687`) |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `POSTGRES_DSN` | Postgres connection string |
