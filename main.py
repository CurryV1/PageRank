import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigs

# --- Input from real-world edge list ---
def load_transition_matrix_from_edge_list(filename):
    graph = nx.DiGraph()

    # Read edges from file
    with open(filename) as f:
        for line in f:
            src, dst = line.strip().split()
            graph.add_edge(src, dst)

    node_labels = sorted(graph.nodes())
    index_map = {node: i for i, node in enumerate(node_labels)}
    size = len(node_labels)
    matrix = np.zeros((size, size))

    # Fill adjacency matrix
    for src, dst in graph.edges():
        j = index_map[src]
        i = index_map[dst]
        matrix[i][j] = 1

    # Normalize each column
    col_sums = matrix.sum(axis=0)
    for j in range(size):
        if col_sums[j] != 0:
            matrix[:, j] /= col_sums[j]

    return matrix, node_labels


# --- Validation ---
def is_valid_transition_matrix(matrix):
    return np.all(matrix >= 0) and np.allclose(np.sum(matrix, axis=0), 1)

def is_probability_vector(vec):
    return np.all(vec >= 0) and np.isclose(np.sum(vec), 1)

# --- Graph Drawing ---
def draw_graph_from_matrix(matrix, labels):
    graph = nx.DiGraph()
    size = matrix.shape[0]

    for i in range(size):
        graph.add_node(labels[i])

    for j in range(size):
        for i in range(size):
            if matrix[i, j] > 0:
                graph.add_edge(labels[j], labels[i], weight=matrix[i, j])

    pos = nx.spring_layout(graph, seed=42)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)
    plt.title('Graph Represented by Transition Matrix')
    plt.show()

# --- Power Iteration ---
def power_iteration(matrix, x0, epsilon=1e-8, max_iter=1000):
    x = x0
    distances = []
    for _ in range(max_iter):
        x_next = matrix @ x
        dist = np.linalg.norm(x_next - x)
        distances.append(dist)
        if dist < epsilon:
            break
        x = x_next
    return x, distances

# --- Load data and execute ---
transition_matrix, node_labels = load_transition_matrix_from_edge_list("real_edges.txt")
start_vector = np.ones(len(transition_matrix)) / len(transition_matrix)

print("Valid matrix:", is_valid_transition_matrix(transition_matrix))
print("Valid start vector:", is_probability_vector(start_vector))

draw_graph_from_matrix(transition_matrix, node_labels)

# --- PageRank Computation ---
steady_state, convergence = power_iteration(transition_matrix, start_vector)

# --- Visualize convergence ---
plt.plot(convergence)
plt.title("Convergence of ||x_n - x_(n+1)||")
plt.xlabel("Iteration")
plt.ylabel("Difference")
plt.grid(True)
plt.show()

# --- Eigenvector Validation ---
eigval, eigvec = eigs(transition_matrix.T, k=1, which='LM')
eigen_rank = eigvec[:, 0].real
eigen_rank = eigen_rank / np.sum(eigen_rank)

print("\nEigenvalue (should be ~1):", eigval[0].real)
print("Dominant eigenvector (normalized):\n", eigen_rank)

# --- Final Outputs ---
print("\nFinal Steady-State Vector x∞:\n", steady_state)
print("\nAx∞:\n", transition_matrix @ steady_state)
print("\nEigenvalues:\n", np.linalg.eigvals(transition_matrix))