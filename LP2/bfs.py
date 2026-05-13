# Breadth First Search (BFS) using Queue
# Undirected Graph Implementation in Python

from collections import deque

class Graph:
    def __init__(self):
        # Dictionary to store adjacency list
        self.graph = {}

    # Function to add an edge
    def add_edge(self, u, v):

        # Add edge u -> v
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

        # Since graph is undirected, add edge v -> u
        if v not in self.graph:
            self.graph[v] = []
        self.graph[v].append(u)

    # BFS Traversal Function
    def bfs(self, start_vertex):

        visited = set()          # To keep track of visited nodes
        queue = deque()          # Queue for BFS

        # Start with the initial vertex
        visited.add(start_vertex)
        queue.append(start_vertex)

        print("BFS Traversal:")

        # Continue until queue becomes empty
        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")

            # Visit all adjacent vertices
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)


# Create graph object
g = Graph()

# Add edges
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

# Perform BFS starting from vertex 0
g.bfs(0)