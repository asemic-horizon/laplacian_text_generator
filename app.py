import streamlit as st
import pandas as pd, numpy as np
import networkx as nx
from scipy.sparse.linalg import lsqr
import random
import joblib
mem = joblib.Memory(location='.', verbose=0)

@st.cache_resource
def generate_ngram_edges(file_path, n=1):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Normalize text and split into words
    words = text[:500_000].split(' ')

    # Generate edges
    edges = []
    for i in range(len(words) - n + 1):
        if i + n < len(words):
            first_ngram = ' '.join(words[i:i+n])
            second_ngram = ' '.join(words[i+1:i+n+1])
            edge = (first_ngram,second_ngram)
            edges.append(edge)

    return edges


@mem.cache
def make_graph(edges):
    G = nx.from_edgelist(edges)
    L = nx.laplacian_matrix(G)
    return G, L

@mem.cache
def solve_for_ith(laplacian, i, noise=1e-3):
    v1 = noise * np.ones(shape=(laplacian.shape[0],1))
    v1[i] = 1
    w = lsqr(laplacian, v1)[0]
    return w

def softmax(x, temperature):
    return np.exp(x / temperature) / (np.exp(x / temperature).sum())

if st.session_state.get('edges') is None:
    st.session_state['edges'] = generate_ngram_edges('atp.txt', n=1)
if st.session_state.get('laplacian') is None:
    G, L = make_graph(st.session_state['edges'])
    st.session_state['graph'] = G
    st.session_state['laplacian'] = L
    term_list = list(G.nodes())
    random.shuffle(term_list)
    st.session_state['term_list'] = term_list
    st.session_state['N'] = st.session_state['laplacian'].shape[0]

c1, c2 = st.columns(2)
term =  c1.selectbox(
    label = 'Type something',
    options = st.session_state['term_list'],
    index = 200,
    placeholder = 'Type something')

def get_code(term):
    return [u for u, v in enumerate(st.session_state['graph'].nodes()) if v == term]

def get_term(ith):
    return [v for u, v in enumerate(st.session_state['graph'].nodes()) if u == ith]

sample_size = c2.slider('Sample size', min_value=1, max_value=1000, value=25)
d1, d2 = st.columns(2)
temperature = d1.slider('Temperature', min_value=0.01, max_value=5.0, value=0.5)
noise = d2.slider('Noise', min_value=0.000, max_value=0.010, value=0.005, step = 1e-4)

e1, e2 = st.columns([1,3])
is_dynamic = e1.checkbox('Dynamic', value = False, disabled=True)
e2.caption("WARNING! Dynamic mode is expensive to compute, slow to report; responds to temperature differently")
if is_dynamic:
    sample = []
    v1 = noise * np.ones(shape=(st.session_state.laplacian.shape[0],1))
    jth = get_code(term)
    for i in range(sample_size):
        v1[jth] = 1
        w = lsqr(st.session_state['laplacian'], v1)[0]
        w = softmax(w, temperature)
        jsample = get_term(jth)
        sample.append(jsample)
        print(jsample)
        jth = np.random.choice(range(st.session_state['N']), p = w.flatten())
else:
    ith = get_code(term)
    w = solve_for_ith(st.session_state['laplacian'], ith, noise)
    w = softmax(w, temperature)
    sample = np.random.choice(st.session_state['graph'].nodes(), size=sample_size, p = w)
st.write(f"{' '.join(sample)}")
st.write("---")
st.caption("FIND ME and follow me through corridors, refectories; to find, you must follow write the electric me on gmailee. ")