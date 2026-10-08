import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

STEP_DIR = Path("step_files")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)


def parse_step(text):
    # Collapse newlines so multi-line entities are handled by a single regex pass
    text = text.replace("\r\n", " ").replace("\n", " ")

    products = {
        m[0]: m[1]
        for m in re.findall(r"#(\d+)\s*=\s*PRODUCT\s*\(\s*'([^']+)'", text, re.IGNORECASE)
    }

    product_defs = {
        m[0]: m[1]
        for m in re.findall(
            r"#(\d+)\s*=\s*PRODUCT_DEFINITION\s*\([^,]*,[^,]*,\s*#(\d+)", text, re.IGNORECASE
        )
    }

    nauos = re.findall(
        r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\([^,]*,[^,]*,[^,]*,\s*#(\d+)\s*,\s*#(\d+)",
        text,
        re.IGNORECASE,
    )

    edges = []
    for parent_pd, child_pd in nauos:
        parent_name = products.get(product_defs.get(parent_pd))
        child_name = products.get(product_defs.get(child_pd))
        if parent_name and child_name and parent_name != child_name:
            edges.append((parent_name, child_name))

    part_names = {name for edge in edges for name in edge}
    return part_names, edges


def ingest(session, parts, edges):
    for name in parts:
        session.run("MERGE (:Part {name: $name})", name=name)
    for parent, child in edges:
        session.run(
            "MATCH (p:Part {name: $parent}), (c:Part {name: $child}) MERGE (p)-[:CONTAINS]->(c)",
            parent=parent,
            child=child,
        )


with driver.session() as session:
    for step_file in STEP_DIR.glob("*.step"):
        text = step_file.read_text(encoding="utf-8", errors="ignore")
        parts, edges = parse_step(text)
        if not parts:
            print(f"No parts found in {step_file.name}")
            continue
        ingest(session, parts, edges)
        print(f"Ingested {len(parts)} parts, {len(edges)} connections from {step_file.name}")

driver.close()
