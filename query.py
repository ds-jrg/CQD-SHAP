from graph import Dataset
import pickle

class Query:
    def __init__(self, query_type: str, query_answer: tuple):
        self.query_type = query_type
        if len(query_answer) != 2:
            raise ValueError("Query answer must be a tuple of (query, answer)")
        elif type(query_answer[1]) is not list:
            raise ValueError("Query answer must be a tuple of (query, answer) where answer is a list")
        self.query = query_answer[0]
        self.answer = query_answer[1]

    def get_query(self):
        return self.query
    
    def get_answer(self):
        return self.answer
    
    def __repr__(self):
        return f"Query(type={self.query_type}, query={self.query}, answer={self.answer})"
    
class QueryDataset:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.queries = {}

    def add_query(self, query_type: str, query_answer: tuple):
        if query_type not in self.queries:
            self.queries[query_type] = []
        query = Query(query_type, query_answer)
        self.queries[query_type].append(query)

    def get_queries(self, query_type: str):
        if query_type not in self.queries:
            raise ValueError(f"No queries of type {query_type} found")
        return self.queries[query_type]
    
    def get_all_queries(self):
        all_queries = []
        for query_type, queries in self.queries.items():
            all_queries.extend(queries)
        return all_queries
    
    def get_num_queries(self):
        return sum(len(queries) for queries in self.queries.values())
    
    def get_num_queries_by_type(self, query_type: str):
        if query_type not in self.queries:
            return 0
        return len(self.queries[query_type])
    
    def load_queries_from_pkl(self, filename: str, query_type: str = ''):
        try:
            with open(filename, 'rb') as f:
                queries = pickle.load(f)
                for query, answer in queries.items():
                    answer = list(answer)
                    self.add_query(query_type, (query, answer))
        except FileNotFoundError:
            raise ValueError(f'File {filename} not found')
        except Exception as e:
            raise ValueError(f'Error loading queries from {filename}: {e}')
        
def human_readable(query: Query, dataset: Dataset):
    if query.query_type == '2p' or query.query_type == '3p':
        anchor = query.query[0][0]
        relations = query.query[0][1]
        rel1 = relations[0]
        rel2 = relations[1]
        if query.query_type =='3p':
            rel3 = relations[2]
        anchor_name = dataset.get_node_by_id(anchor)
        rel1_name = dataset.get_relation_by_id(rel1)
        rel2_name = dataset.get_relation_by_id(rel2)
        if query.query_type == '3p':
            rel3_name = dataset.get_relation_by_id(rel3)
        anchor_title = dataset.get_title_by_node(anchor_name)
        answers_titles = [dataset.get_title_by_node(dataset.get_node_by_id(a)) for a in query.answer]
        if query.query_type == '3p':
            print(f"Query:\n{anchor_title}\t--{rel1_name}-->\tV1")
            print(f"V1\t--{rel2_name}-->\tV2")
            print(f"V2\t--{rel3_name}-->\t?")
        else:
            print(f"Query:\n{anchor_title}\t--{rel1_name}-->\tV")
            print(f"V\t--{rel2_name}-->\t?")
        print(f"\nAnswer Set (?): \n{answers_titles}")
    elif query.query_type == '2u' or query.query_type == '2i' or query.query_type == '3i':
        query1 = query.query[0]
        query2 = query.query[1]
        anchor1 = query1[0]
        relation1 = query1[1][0]
        anchor2 = query2[0]
        relation2 = query2[1][0]
        anchor1_name = dataset.get_node_by_id(anchor1)
        anchor2_name = dataset.get_node_by_id(anchor2)
        rel1_name = dataset.get_relation_by_id(relation1)
        rel2_name = dataset.get_relation_by_id(relation2)
        anchor1_title = dataset.get_title_by_node(anchor1_name)
        anchor2_title = dataset.get_title_by_node(anchor2_name)
        answers_titles = [dataset.get_title_by_node(dataset.get_node_by_id(a)) for a in query.answer]
        print(f"Query:\n{anchor1_title}\t--{rel1_name}-->\tV1")
        print(f"{anchor2_title}\t--{rel2_name}-->\tV2")
        if query.query_type == '3i':
            query3 = query.query[2]
            anchor3 = query3[0]
            relation3 = query3[1][0]
            anchor3_name = dataset.get_node_by_id(anchor3)
            rel3_name = dataset.get_relation_by_id(relation3)
            anchor3_title = dataset.get_title_by_node(anchor3_name)
            print(f"{anchor3_title}\t--{rel3_name}-->\tV3")
        if query.query_type == '2u':
            print(f"V1\tOR\tV2\t-->\t?")
        elif query.query_type == '2i':
            print(f"V1\tAND\tV2\t-->\t?")
        elif query.query_type == '3i':
            print(f"V1\tAND\tV2\tAND\tV3\t-->\t?")
        print(f"\nAnswer Set (?): \n{answers_titles}")
    elif query.query_type == 'up' or query.query_type == 'ip':
        query1 = query.query[0]
        anchor1 = query1[0]
        relation1 = query1[1][0]
        query2 = query.query[1]
        anchor2 = query2[0]
        relation2 = query2[1][0]
        relation3 = query.query[2]
        anchor1_name = dataset.get_node_by_id(anchor1)
        anchor2_name = dataset.get_node_by_id(anchor2)
        rel1_name = dataset.get_relation_by_id(relation1)
        rel2_name = dataset.get_relation_by_id(relation2)
        rel3_name = dataset.get_relation_by_id(relation3)
        anchor1_title = dataset.get_title_by_node(anchor1_name)
        anchor2_title = dataset.get_title_by_node(anchor2_name)
        answers_titles = [dataset.get_title_by_node(dataset.get_node_by_id(a)) for a in query.answer]
        print(f"Query:\n{anchor1_title}\t--{rel1_name}-->\tV1")
        print(f"{anchor2_title}\t--{rel2_name}-->\tV2")
        if query.query_type == 'up':
            print(f"V1\tOR\tV2\t-->\tV3")
        elif query.query_type == 'ip':
            print(f"V1\tAND\tV2\t-->\tV3")
        print(f"V3\t--{rel3_name}-->\t?")
        print(f"\nAnswer Set (?): \n{answers_titles}")
    elif query.query_type == 'pi':
        branch1 = query.query[0]
        branch2 = query.query[1]
        anchor1 = branch1[0]
        relation1 = branch1[1][0]
        relation2 = branch1[1][1]
        anchor2 = branch2[0]
        relation3 = branch2[1][0]
        anchor1_name = dataset.get_node_by_id(anchor1)
        anchor2_name = dataset.get_node_by_id(anchor2)
        rel1_name = dataset.get_relation_by_id(relation1)
        rel2_name = dataset.get_relation_by_id(relation2)
        rel3_name = dataset.get_relation_by_id(relation3)
        anchor1_title = dataset.get_title_by_node(anchor1_name)
        anchor2_title = dataset.get_title_by_node(anchor2_name)
        answers_titles = [dataset.get_title_by_node(dataset.get_node_by_id(a)) for a in query.answer]
        print(f"Query:\n{anchor1_title}\t--{rel1_name}-->\tV1")
        print(f"V1\t--{rel2_name}-->\tV2")
        print(f"{anchor2_title}\t--{rel3_name}-->\tV3")
        print(f"V2\tAND\tV3\t-->\t?")
        print(f"\nAnswer Set (?): \n{answers_titles}")
        
        