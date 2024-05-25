import numpy as np
import networkx as nx


def softmax(x, temperature):
    return np.exp(x / temperature) / (np.exp(x / temperature).sum())


def dict_temperature_sample(scores: dict, temperature=1):
    """Temperature sampling from a key-value pair dictionary"""
    nodes, scores = np.array(list(scores.keys())), np.array(list(scores.values()))
    sampled_node = np.random.choice(nodes, p=softmax(scores, temperature))
    return sampled_node


def sample_ego_graph(graph,
                     node,
                     radius=2,
                     temperature=1,
                     scoring_function=nx.betweenness_centrality,
                     global_score=False):
    """Sample an ego subgraph from the graph centered on the node
    with a probability proportional to the centrality of the nodes"""
    ego_graph = nx.ego_graph(graph, node, radius=radius)
    centrality_score = scoring_function(ego_graph)
    if not global_score:
        centrality_score = {
            k: v
            for k, v in centrality_score.items() if k in ego_graph.nodes()
        }
    return dict_temperature_sample(centrality_score, temperature)


def levenshtein(s1, s2):
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
def closest_term(term, term_list):
    return min(term_list, key=lambda x: levenshtein(term, x))