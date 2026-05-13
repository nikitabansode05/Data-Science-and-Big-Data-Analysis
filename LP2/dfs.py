# Depth First Search (DFS) using Recursion
# Undirected Graph Implementation in Python

class Graph:
    def __init__(self):
        # Dictionary to store adjacency list
        self.graph = {}

    # Function to add an edge to the graph
    def add_edge(self, u, v):
        # Add edge from u to v
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

        # Since graph is undirected, add edge from v to u
        if v not in self.graph:
            self.graph[v] = []
        self.graph[v].append(u)

    # Recursive DFS function
    def dfs_recursive(self, vertex, visited):
        # Mark current node as visited
        visited.add(vertex)

        # Print the visited vertex
        print(vertex, end=" ")

        # Visit all adjacent vertices
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                self.dfs_recursive(neighbor, visited)

    # Function to start DFS traversal
    def dfs(self, start_vertex):
        visited = set()
        print("DFS Traversal:")
        self.dfs_recursive(start_vertex, visited)


# Create graph object
g = Graph()

# Add edges
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

# Perform DFS starting from vertex 0
g.dfs(0)