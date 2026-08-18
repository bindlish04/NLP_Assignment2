"""
Visualization helpers for Task 1 (ontology) and Task 2 (knowledge graph).

Figures are saved to outputs/ and also displayed inline in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless / batch execution
import matplotlib.pyplot as plt
import networkx as nx

# Avoid drawing huge hairballs in subgraph plots.
MAX_SUBGRAPH_NODES = 40


def draw_ontology_schema(ontology, output_path: Path) -> None:
    """Draw entity classes as nodes and relationship types as labeled edges."""
    graph = nx.DiGraph()
    for cls in ontology.entity_classes:
        graph.add_node(cls, kind="class")

    for rel in ontology.relationship_types.values():
        for dom in (part.strip() for part in rel.domain.split("|")):
            for rng in (part.strip() for part in rel.range.split("|")):
                if dom in graph and rng in graph:
                    graph.add_edge(dom, rng, label=rel.name)

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(graph, pos, node_color="#87CEEB", node_size=2200)
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=16)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "label"), font_size=7)
    plt.title(f"Ontology Schema - {ontology.domain}")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_subgraph(
    graph: nx.MultiDiGraph,
    center_node: str,
    output_path: Path,
    radius: int = 1,
    max_nodes: int = MAX_SUBGRAPH_NODES,
) -> None:
    """Draw an ego network around center_node, capped for readability."""
    if center_node not in graph:
        return

    ego = nx.ego_graph(graph, center_node, radius=radius)
    if ego.number_of_nodes() == 0:
        return

    # If the ego graph is large, keep center + highest-degree neighbors only.
    if ego.number_of_nodes() > max_nodes:
        neighbors = sorted(
            ((n, ego.degree(n)) for n in ego.nodes() if n != center_node),
            key=lambda x: x[1],
            reverse=True,
        )
        keep = {center_node, *(n for n, _ in neighbors[: max_nodes - 1])}
        sub = graph.subgraph(keep).copy()
    else:
        sub = graph.subgraph(ego.nodes()).copy()

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(sub, seed=7)
    colors = ["#FFB347" if n == center_node else "#98FB98" for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=900)
    nx.draw_networkx_labels(sub, pos, font_size=7)
    nx.draw_networkx_edges(sub, pos, arrows=True, arrowsize=12)
    plt.title(f"Knowledge Graph Subgraph - {center_node.title()}")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
