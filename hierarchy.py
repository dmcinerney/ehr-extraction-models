import copy

class Hierarchy:
    # ASSUMES THE GRAPH IS A DAG WITH ONLY ONE NODE OF IN_DEGREE 0!
    @classmethod
    def from_graph(cls, graph):
        max_index = -1
        indices = {}
        options = {}
        parents = {}
        descriptions = {}
        for n in graph.nodes:
            if graph.in_degree(n) == 0:
                start = n
            options[n] = []
            for i,succ in enumerate(graph.successors(n)):
                indices[succ] = i
                options[n].append(succ)
                max_index = max(max_index, i)
                if succ in parents.keys():
                    raise Exception
                parents[succ] = n
                descriptions[succ] = graph.nodes[succ]['description']\
                    if graph.nodes[succ]['description'] is not None else ''
        indices['max_index'] = max_index
        return cls(start, options, indices, parents, descriptions)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(dictionary['start'],
                   dictionary['options'],
                   dictionary['indices'],
                   dictionary['parents'],
                   dictionary['descriptions'])

    def __init__(self, start, options, indices, parents, descriptions):
        self.start = start
        self.options = options
        self.indices = indices
        self.parents = parents
        self.descriptions = descriptions

    def ancestors(self, nodes, stop_nodes=set()):
        node_stack = copy.deepcopy(nodes)
        new_nodes = set()
        while len(node_stack) > 0:
            node = node_stack.pop()
            if node in stop_nodes: continue # don't add stop nodes
            if node in new_nodes: continue # don't add nodes already there
            if node == self.start: continue # don't add the start node
            new_nodes.add(node)
            node_stack.extend([self.parents[node]])
        return list(new_nodes)

    def get_descriptions(self, nodes):
        return [self.descriptions[node] for node in nodes]

    def linearize(self, node):
        backwards_options = []
        while node != self.start:
            backwards_options.append(self.indices[node])
            node = self.parents[node]
        return list(reversed(backwards_options))

    def path(self, node):
        backwards_path = []
        while node != self.start:
            backwards_path.append(node)
            node = self.parents[node]
        return list(reversed(backwards_path))

    def depth(self, node):
        depth = 0
        while node != self.start:
            depth += 1
            node = self.parents[node]
        return depth

    def delinearize(self, linearized_node):
        node = self.start_node
        for i in linearized_node:
            node = self.options[node][i]
        return node

    def to_dict(self):
        return {"start":self.start,
                "options":self.options,
                "indices":self.indices,
                "parents":self.parents,
                "descriptions":self.descriptions}
