import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import nodes
from scipy.sparse.linalg import eigs

# Define the transition matrix A

# test values (hardcoded)
# A = np.array([
#     [0,   1,   0],
#     [0.5, 0,   0.5],
#     [0.5, 0,   0.5]
# ])

# user input matrix A and vectorx0 from file
# A = np.loadtxt("matrix.txt")
# x0 = np.loadtxt("x0.txt")




# Define the starting vector x0

# test values (hardcoded)
# x0 = np.array([0.5, 0.3, 0.2])

# normalize to a probability vector
# x0 = x0 / np.sum(x0)

# Validate transition matrix and starting vector ---
def is_valid_transition_matrix(matrix):
    return np.all(matrix >= 0) and np.allclose(np.sum(matrix, axis=0), 1)

def is_probability_vector(vec):
    return np.all(vec >= 0) and np.isclose(np.sum(vec), 1)

def load_transition_matrix_from_edge_list(filename):
    G = nx.DiGraph()

    # read edges
    with open(filename) as f:
        for line in f:
            src, dst = line.strip().split()
            G.add_edge(src, dst)

    nodes = sorted(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    A = np.zeros((n, n))

    # Fill adjacency matrix
    for src, dst in G.edges():
        j = node_indices[src]
        i = node_indices[dst]
        A[i][j] = 1

    # Normalize each column to get transition probabilities
    col_sums = A.sum(axis=0)
    for j in range(n):
        if col_sums[j] != 0:
            A[:, j] /= col_sums[j]

        return A, nodes

A, labels = load_transition_matrix_from_edge_list("real_edges.txt")
x0 = np.ones(len(A)) / len(A)


def draw_graph_from_matrix(A, labels):
    n = A.shape[0]
    G = nx.DiGraph()

    # Add nodes
    for i in range(n):
        G.add_node(labels[i])

    # Add edges -> only where A[i, j] > 0
    for j in range(n):
        for i in range(n):
            if A[i, j] > 0:
                G.add_edge(labels[j], labels[i], weight=A[i, j]) # edge from j to i

    # Draw the graph
    pos = nx.spring_layout(G, seed=42) # deterministic layout
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Graph Represented  by Transition Matrix A')
    plt.show()


print("Valid matrix:", is_valid_transition_matrix(A))
print("Valid start vector:", is_probability_vector(x0))
# visualize matrix as a graph
draw_graph_from_matrix(A, labels)

# Power iteration to compute steady state ---
def power_iteration(A, x0, epsilon=1e-8, max_iter=1000):
    x = x0
    distances = []
    for _ in range(max_iter):
        x_next = A @ x
        dist = np.linalg.norm(x_next - x)
        distances.append(dist)
        if dist < epsilon:
            break
        x = x_next
    return x, distances

x_inf, convergence = power_iteration(A, x0)

# Eigenvalues and eigenvectors ---
eigenvalues, eigenvectors = np.linalg.eig(A)

# Plot convergence ---
plt.plot(convergence)
plt.title("Convergence of ||x_n - x_(n+1)||")
plt.xlabel("Iteration")
plt.ylabel("Difference")
plt.grid(True)
plt.show()

# Transpose A -> eigs expects left eigenvectors for Ax = x
eigenvalue, eigenvector = eigs(A.T, k=1, which='LM') # 'LM' = Largest Magnitude

# Normalize the eigenvector to be a probability vector
ev = eigenvector[:, 0].real
ev = ev / np.sum(ev)

print("\nEigenvalue (should be ~1): ", eigenvalue[0].real)
print("Dominant eigenvector (normalized):\n", ev)

# Display results ---
print("\nFinal Steady-State Vector x∞:\n", x_inf)
print("\nAx∞:\n", A @ x_inf)
print("\nEigenvalues:\n", eigenvalues)