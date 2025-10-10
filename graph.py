import pickle
import numpy as np

class Dataset:
    def __init__(self):
        self.id2node = {}
        self.node2id = {}
        self.id2rel = {}
        self.rel2id = {}
        self.node2title = {}
        self.title2node = {}

    def load_key_value_files(self, filename):
        '''
        Load key-value pairs from a file
        Args:
            filename (str): The name of the file to load.
            file_format (str): The format of the file ('pkl' or 'txt').
        Returns:
            tuple: (id2value, value2id) where:
                id2value (dict): Dictionary mapping IDs to values.
                value2id (dict): Dictionary mapping values to IDs.
        '''
        id2value = {}
        value2id = {}
        file_format = filename.split('.')[-1]
        if file_format == 'pkl':
            with open(filename, 'rb') as f:
                id2value = pickle.load(f)
        elif file_format == 'txt':
            with open(filename, 'r') as f:
                id2value = {}
                for line in f:
                    id, value = line.strip().split('\t')
                    id2value[id] = value
        else:
            raise ValueError("Unsupported file format. Use 'pkl' or 'txt'.")
        value2id = {v: k for k, v in id2value.items()}
        return id2value, value2id
    
    def set_id2node(self, filename):
        id2node, node2id = self.load_key_value_files(filename)
        self.id2node = id2node
        self.node2id = node2id
        print(f"Loaded {len(self.id2node)} nodes from {filename}.")

    def get_node_by_id(self, node_id):
        return self.id2node.get(node_id, None)
    
    def get_id_by_node(self, node):
        return self.node2id.get(node, None)

    def set_id2rel(self, filename):
        id2rel, rel2id = self.load_key_value_files(filename)
        self.id2rel = id2rel
        self.rel2id = rel2id
        print(f"Loaded {len(self.id2rel)} relations from {filename}.")

    def get_relation_by_id(self, rel_id):
        return self.id2rel.get(rel_id, None)
    
    def get_id_by_relation(self, relation):
        return self.rel2id.get(relation, None)

    def set_node2title(self, filename):
        try:
            node2title, title2node = self.load_key_value_files(filename)
        except:
            print(f"Failed to load node titles from {filename}. Using entity names as titles.")
            node2title = {v: v for k, v in self.id2node.items()}
            title2node = {v: v for k, v in self.id2node.items()}
        self.node2title = node2title
        self.title2node = title2node
        print(f"Loaded {len(self.node2title)} node titles from {filename}.")


    def get_title_by_node(self, node):
        return self.node2title.get(node, None)
    
    def get_node_by_title(self, title):
        return self.title2node.get(title, None)
    
    def get_num_nodes(self):
        return len(self.id2node)
    
    def get_title_by_id(self, node_id):
        node = self.get_node_by_id(node_id)
        if node is not None:
            return self.get_title_by_node(node)
        return None

class Node:
    def __init__(self, name: str, id: int, title: str):
        self.id = id
        self.name = name
        self.title = title

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name
    
    def get_title(self):
        return self.title

class Edge:
    def __init__(self, name:str, id: int, head: Node, tail: Node):
        self.id = id
        self.name = name
        self.head = head
        self.tail = tail

    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

class Graph:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.edges = []
    
    def add_edge(self, head: str, relation: str, tail: str, skip_missing: bool = True, add_reverse: bool = True):
        head_id = self.dataset.get_id_by_node(head)
        tail_id = self.dataset.get_id_by_node(tail)
        relation_id = self.dataset.get_id_by_relation(relation)
        skipped = 0
        if head_id is None and skip_missing:
            # print(f'Node {head} not found in dataset, skipping edge')
            skipped += 1
        elif tail_id is None and skip_missing:
            # print(f'Node {tail} not found in dataset, skipping edge')
            skipped += 1
        elif relation_id is None and skip_missing:
            # print(f'Relation {relation} not found in dataset, skipping edge')
            skipped += 1
        else:
            head_node = Node(head, head_id, self.dataset.get_title_by_node(head))
            tail_node = Node(tail, tail_id, self.dataset.get_title_by_node(tail))
            edge = Edge(relation, relation_id, head_node, tail_node)
            self.edges.append(edge)
            if add_reverse:
                reverse_relation = f'{relation}_reverse'
                reverse_relation_id = self.dataset.get_id_by_relation(reverse_relation)
                reverse_edge = Edge(reverse_relation, reverse_relation_id, tail_node, head_node)
                self.edges.append(reverse_edge)
        return skipped

    def load_triples(self, filename: str, skip_missing: bool = True, add_reverse: bool = True):
        try:
            counter = 0
            with open(filename, 'r') as f:
                for line in f:
                    head, relation, tail = line.strip().split('\t')
                    counter += self.add_edge(head, relation, tail, skip_missing, add_reverse)
            print(f'Loaded {len(self.edges)} edges from {filename}, skipped {counter} edges due to missing nodes or relations.')
        except FileNotFoundError:
            raise ValueError(f'File {filename} not found')
        except Exception as e:
            raise ValueError(f'Error loading triples from {filename}: {e}')
        
    def get_num_edges(self):
        return len(self.edges)
    
    def get_edges(self):
        return self.edges

    def get_num_nodes(self):
        return self.dataset.get_num_nodes()