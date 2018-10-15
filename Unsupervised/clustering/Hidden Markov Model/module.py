import numpy as np
import pandas as pd
import networkx.drawing.nx_pydot as gl
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint

##matplotlib inline

# create state space and initial state probabilities

states = ['O1', 'O2', 'O3']
pi = [0.25, 0.4, 0.35]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.4, 0.2, 0.4]
q_df.loc[states[1]] = [0.45, 0.45, 0.1]
q_df.loc[states[2]] = [0.45, 0.25, .3]

print(q_df)

q = q_df.values
print('\n')
print(q, q.shape)
print('\n')
print(q_df.sum(axis=1))

from pprint import pprint


# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print('Nodes:\n')
print(G.nodes())
print('\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print('Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'markov.dot')
plt.show()

# create state space and initial state probabilities

hidden_states = ['S1', 'S2']
pi = [0.5, 0.5]
print('\n')
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n')
print(state_space.sum())

# create hidden transition matrix
# a or alpha
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n')
print(a)
print(a.shape)
print('\n')
print(a_df.sum(axis=1))

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states
# and O is number of different possible observations

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

b = b_df.values
print('\n')
print(b)
print(b.shape)
print('\n')
print(b_df.sum(axis=1))

# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
print('Nodes:\n')
print(G.nodes())
print('\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

print('Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)
plt.show()

# create edge labels
emit_edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=emit_edge_labels)
# plt.show()
nx.drawing.nx_pydot.write_dot(G, 'hidden_markov.dot')

# observation sequence of dog's behaviors
# observations are encoded numerically

obs_map = {'O1': 0, 'O2': 1, 'O3': 2}
obs = np.array([1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1])

inv_obs_map = dict((v, k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print(pd.DataFrame(np.column_stack([obs, obs_seq]),
                   columns=['Obs_code', 'Obs_seq']))


def viterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    path = np.zeros(T)
    delta = np.zeros((nStates, T))
    phi = np.zeros((nStates, T))

    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])

    path[T - 1] = np.argmax(delta[:, T - 1])
    for t in range(T - 2, -1, -1):
        # path[t] = phi[int(path[t+1]): int(t+1) , int(t+1)]
        path[t] = phi[int(path[t + 1]), int(t + 1)]

    return path, delta, phi


path, delta, phi = viterbi(pi, a, b, obs)
print('\n')
print('single best state path: ', path)
print('delta:\n', delta)
print('phi:\n', phi)

state_map = {0: 'S1', 1: 'S2'}
state_path = [state_map[v] for v in path]

result = (pd.DataFrame()
          .assign(Observation=obs_seq)
          .assign(Best_Path=state_path))

print(result)