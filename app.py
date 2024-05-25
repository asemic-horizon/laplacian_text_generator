import streamlit as st
import pandas as pd, numpy as np
import networkx as nx
import random
import joblib
import sampler
from itertools import combinations

mem = joblib.Memory(location='.', verbose=0)
centrality = mem.cache(nx.betweenness_centrality)


def split_in_nchars(txt, nchars):
    """This function splits a string in contiguous strings of nchars
    e.g. split_in_nchars("hello world",3) -> ['hel', 'llo", " wo", "rld']"""
    return [txt[i:i + nchars] for i in range(0, len(txt), nchars)]


@mem.cache
def generate_ngram_edges(file_path, n=3):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Normalize text and split into words
    lines = text.split('\n')
    random.shuffle(lines)
    edges = []
    for line in lines:
       words = line.split(' ')
       edges += list(combinations(words, 2))
       if len(edges) > 10_000:
           break
    return edges


@mem.cache
def make_graph(edges):
    G = nx.from_edgelist(edges)
    return G

    return np.exp(x / temperature) / (np.exp(x / temperature).sum())

explanation = "This text generator was conceived as an exploration of"
explanation += " temperature sampling. In lieu of a fancy attention transformer"
explanation += " model, it generates a graph from text by connecting all words"
explanation += " that within a paragraph, thus allowing paragraphs with coinciding"
explanation += " words to be connected to a second deegree."
explanation += " A sampler then samples the next word"
explanation += " so that words that appear in more than one paragraph are more likely"
explanation += " to be sampled. The temperature parameter does what it does in"
explanation += " large language models."
with st.expander("Click for explanations. Maybe better don't.", expanded = False):
    st.caption(explanation)

with st.spinner("This computation will run only once, please wait."):
    edges = generate_ngram_edges('atp.txt')
    print(f'Number of edges: {len(edges)}')
    G = make_graph(edges)
term_list = list(G.nodes())
random.shuffle(term_list)

with st.form(key='my_form'):
    term = st.text_input(label='Type something',
                        placeholder='something short is best')

    d1, d2, d3, d4 = st.columns(4)
    sample_size = d3.slider('Sample size', min_value=1, max_value=1000, value=75)
    temperature = d1.slider('Temperature',
                            min_value=0.01,
                            max_value=5.0,
                            value=1.25)
    radius = d2.slider('Radius', min_value=1, max_value=10, value=2)
    global_score = d4.checkbox('Global score', value=False)
    submit_button = st.form_submit_button(label='Answer me!')

f1, f2, f3 = st.columns(3)
f2.write(' '.join(term))
term = sampler.closest_term(term, term_list)
f2.write(' '.join(term))

sample = [term]
if submit_button:
    f2.write(f"{' '.join(term)}")
    while len(sample)<=sample_size:
        next_term = sampler.sample_ego_graph(graph=G,
                                            node=term,
                                            radius=radius,
                                            temperature=temperature,
                                            scoring_function=centrality,
                                            global_score=global_score)
        print(next_term)
        f2.write(f"{' '.join(next_term)}")
        sample.append(next_term)
        term = next_term
st.write("---")
st.caption(
    "FIND ME and follow me through corridors, refectories; to find, you must follow write the electric me on gmailee. "
)
