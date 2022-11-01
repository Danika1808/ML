from queue import PriorityQueue
from random import randint, uniform
import numpy as np
import networkx as nx
from matplotlib import animation, rc
import matplotlib.pyplot as plt

rc('animation', html='html5')

NUM_NODES = 8


def random_node():
    return randint(0, NUM_NODES - 1)


def random_weight():
    return np.random.randint(1, 6)


graph = nx.Graph()

for i in range(1, NUM_NODES):
    graph.add_edge(i - 1, i, weight=random_weight())

for _ in range(NUM_NODES * 4):
    graph.add_edge(
        random_node(), random_node(), weight=random_weight()
    )

pos = nx.circular_layout(graph)
labels = nx.get_edge_attributes(graph, "weight")

all_edges = set(
    tuple(sorted((n1, n2))) for n1, n2 in graph.edges()
)
edges_in_mst = []
nodes_on_mst = []
edge_weights = []

fig, ax = plt.subplots(figsize=(6, 4))


def prims():
    pqueue = PriorityQueue()
    start_node = random_node()
    for neighbor in graph.neighbors(start_node):
        edge_data = graph.get_edge_data(start_node, neighbor)
        edge_weight = edge_data["weight"]
        pqueue.put((edge_weight, (start_node, neighbor)))
    while len(nodes_on_mst) < NUM_NODES:
        _, edge = pqueue.get(pqueue)

        if edge[0] not in nodes_on_mst:
            new_node = edge[0]
        elif edge[1] not in nodes_on_mst:
            new_node = edge[1]
        else:
            continue
        for neighbor in graph.neighbors(new_node):
            edge_data = graph.get_edge_data(new_node, neighbor)
            edge_weight = edge_data["weight"]
            pqueue.put((edge_weight, (new_node, neighbor)))
        edges_in_mst.append(tuple(sorted(edge)))
        nodes_on_mst.append(new_node)
        edge_weights.append(edge_weight)
        yield edges_in_mst


def update(mst_edges):
    ax.clear()
    nx.draw_networkx_nodes(graph, pos, node_size=25, ax=ax)
    nx.draw_networkx_edges(
        graph, pos, edgelist=all_edges, alpha=0.1,
        edge_color='g', width=1, ax=ax
    )
    nx.draw_networkx_edge_labels(graph,
                                 pos=pos,
                                 edge_labels=labels,
                                 font_color='red',
                                 label_pos=0.8,
                                 bbox=dict(
                                     facecolor='none',
                                     edgecolor='none', )
                                 )
    nx.draw_networkx_edges(
        graph, pos, edgelist=mst_edges, alpha=1.0,
        edge_color='b', width=1, ax=ax
    )


def do_nothing():
    # FuncAnimation requires an initialization function. We don't
    # do any initialization, so we provide a no-op function.
    pass


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=do_nothing,
    frames=prims,
    interval=500,
)

plt.show()
count_clusters = 3
edges = edges_in_mst
for i in range(0, len(edge_weights) - 1):
    for j in range(len(edge_weights) - 1):
        if edge_weights[j] > edge_weights[j + 1]:
            temp = edge_weights[j]
            edge_weights[j] = edge_weights[j + 1]
            edge_weights[j + 1] = temp
            temp2 = edges[j]
            edges[j] = edges[j + 1]
            edges[j + 1] = temp2

new_edge = []
new_weight = []
for edge, weight in list(zip(edges, edge_weights))[:len(edges) - count_clusters + 1]:
    new_edge.append(edge)
    new_weight.append(weight)

plt.figure()
ax = plt.gca()
nx.draw_networkx_nodes(graph, pos, node_size=25, ax=ax)
nx.draw_networkx_edges(
    graph, pos, edgelist=new_edge, alpha=1.0,
    edge_color='b', width=1, ax=ax
)
plt.show()
