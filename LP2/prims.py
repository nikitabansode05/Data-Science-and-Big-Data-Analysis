# Prim's Minimum Spanning Tree (MST) Algorithm
# Corrected Python Program

import heapq

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = {i: [] for i in range(vertices)}

    # Function to add edges
    def add_edge(self, u, v, weight):
        self.graph[u].append((weight, v))
        self.graph[v].append((weight, u))   # Undirected Graph

    # Prim's Algorithm
    def prim_mst(self):

        visited = set()

        # Min Heap -> (weight, current_vertex, parent)
        min_heap = [(0, 0, -1)]

        total_cost = 0
        mst_edges = []

        while min_heap and len(visited) < self.V:

            weight, u, parent = heapq.heappop(min_heap)

            # Skip if already visited
            if u in visited:
                continue

            visited.add(u)
            total_cost += weight

            # Ignore first node parent = -1
            if parent != -1:
                mst_edges.append((parent, u, weight))

            # Visit neighbors
            for edge_weight, v in self.graph[u]:

                if v not in visited:
                    heapq.heappush(min_heap, (edge_weight, v, u))

        # Print MST
        print("Edges in Minimum Spanning Tree:")

        for u, v, weight in mst_edges:
            print(f"{u} -- {v} == {weight}")

        print("Total Cost of MST =", total_cost)


# Driver Code
g = Graph(5)

# Add edges
g.add_edge(0, 1, 2)
g.add_edge(0, 3, 6)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 8)
g.add_edge(1, 4, 5)
g.add_edge(2, 4, 7)
g.add_edge(3, 4, 9)

# Run Prim's Algorithm
g.prim_mst()