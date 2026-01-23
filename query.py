from graph import Dataset
import pickle

def query_v2_to_v1(query_v2: tuple, query_type: str):
    '''
    The query structures in the second benchmark (is complex query answering really complex?) are different from the original CQD format in some cases.
    This function converts a query from v2 format to v1 format based on the query type, if needed.
    '''
    if query_type in ["2i", "3i", "4i", "pi"]:
        return query_v2  # these types have the same format in v1 and v2
    elif query_type == "2u":
        query_v1 = query_v2[:-1]
    elif query_type == "1p":
        anchor = query_v2[0]
        relation1, = query_v2[1]
        query_v1 = ((anchor, (relation1,)),)
    elif query_type == "2p":
        anchor = query_v2[0]
        relation1, relation2 = query_v2[1]
        query_v1 = ((anchor, (relation1, relation2)),)
    elif query_type == "3p":
        anchor = query_v2[0]
        relation1, relation2, relation3 = query_v2[1]
        query_v1 = ((anchor, (relation1, relation2, relation3)),)
    elif query_type == "4p":
        anchor = query_v2[0]
        relation1, relation2, relation3, relation4 = query_v2[1]
        query_v1 = ((anchor, (relation1, relation2, relation3, relation4)),)
    elif query_type == "ip":
        # v2: ((('e', ('r',)), ('e', ('r',))), ('r',))
        # v1: (('e', ('r',)), ('e', ('r',)), 'r')
        branch1 = query_v2[0]
        branch2 = query_v2[1]
        atom1 = branch1[0]
        atom2 = branch1[1]
        anchor1 = atom1[0]
        relation1, = atom1[1]
        anchor2 = atom2[0]
        relation2, = atom2[1]
        relation3 = branch2[0]
        query_v1 = ((anchor1, (relation1,)), (anchor2, (relation2,)), relation3)
    elif query_type == "up":
        # v2: ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
        # v1: (('e', ('r',)), ('e', ('r',)), 'r')
        branch1 = query_v2[0]
        atom1 = branch1[0]
        atom2 = branch1[1]
        anchor1 = atom1[0]
        relation1, = atom1[1]
        anchor2 = atom2[0]
        relation2, = atom2[1]
        relation3 = query_v2[1][0]
        query_v1 = ((anchor1, (relation1,)), (anchor2, (relation2,)), relation3)
    else:
        raise ValueError(f"Unknown query type: {query_type}")
    
    return query_v1

def extract_atoms_from_query(query: tuple, query_type: str):
    '''
    Extract the atoms from a Query object and return them as a dictionary.
    Each item in the list is a dictionary with keys: 'head', 'relation', 'tail'.
    '''
    
    atoms = {}
    
    if query_type in ['1p', '2p', '3p', '4p']:
        anchor = query[0][0]
        relations = query[0][1]
        atom0 = {'head': anchor, 'relation': relations[0], 'tail': 'V1'}
        atoms[0] = atom0
        for i in range(1, len(relations)):
            atom = {'head': f'V{i}', 'relation': relations[i], 'tail': f'V{i+1}'}
            atoms[i] = atom
    elif query_type in ['2i', '3i', '4i']:
        for j in range(len(query)):
            branch = query[j]
            anchor = branch[0]
            relation, = branch[1]
            atom = {'head': anchor, 'relation': relation, 'tail': 'V1'}
            atoms[j] = atom
    elif query_type == '2u':
        for j in range(2):
            branch = query[j]
            anchor = branch[0]
            relation, = branch[1]
            atom = {'head': anchor, 'relation': relation, 'tail': 'V1'}
            atoms[j] = atom
    elif query_type == 'up':
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        atom_0 = {'head': anchor1, 'relation': relation1, 'tail': 'V1'}
        atom_1 = {'head': anchor2, 'relation': relation2, 'tail': 'V1'}
        atom_2 = {'head': 'V1', 'relation': relation3, 'tail': 'V2'}
        atoms[0] = atom_0
        atoms[1] = atom_1
        atoms[2] = atom_2
    elif query_type == 'ip':
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        atom_0 = {'head': anchor1, 'relation': relation1, 'tail': 'V1'}
        atom_1 = {'head': anchor2, 'relation': relation2, 'tail': 'V1'}
        atom_2 = {'head': 'V1', 'relation': relation3, 'tail': 'V2'}
        atoms[0] = atom_0
        atoms[1] = atom_1
        atoms[2] = atom_2
    elif query_type == 'pi':
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1, relation2 = atom1[1]
        relation3 = atom2[1][0]
        atom_0 = {'head': anchor1, 'relation': relation1, 'tail': 'V1'}
        atom_1 = {'head': 'V1', 'relation': relation2, 'tail': 'V2'}
        atom_2 = {'head': anchor2, 'relation': relation3, 'tail': 'V2'}
        atoms[0] = atom_0
        atoms[1] = atom_1
        atoms[2] = atom_2
    else:
        raise ValueError(f"Unknown query type: {query_type}")
    
    return atoms
    
        
class Query:
    def __init__(self, query_type: str, query_answer: tuple):
        self.query_type = query_type
        if len(query_answer) != 2:
            raise ValueError("Query answer must be a tuple of (query, answer)")
        elif type(query_answer[1]) is not list:
            raise ValueError("Query answer must be a tuple of (query, answer) where answer is a list")
        self.query = query_answer[0]
        self.answer = query_answer[1]
        self.atoms = extract_atoms_from_query(self.query, self.query_type)
        
    def get_type(self):
        return self.query_type

    def get_query(self):
        return self.query
    
    def get_answer(self):
        return self.answer
    
    def get_atoms(self):
        return self.atoms
    
    def get_num_atoms(self):
        return len(self.atoms)
    
    def get_atom(self, index: int):
        if index < 0 or index >= self.get_num_atoms():
            raise IndexError("Atom index out of range")
        return self.atoms[index]
    
    def __repr__(self):
        return f"Query(type={self.query_type}, query={self.query}, answer={self.answer})"
    
class QueryDataset:
    def __init__(self, dataset: Dataset, type: str = 'complete'):
        self.dataset = dataset
        self.queries = {}
        self.query_structure_to_type = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                }
        self.query_type_to_structure = {value: key for key, value in self.query_structure_to_type.items()}
        assert type in ['complete', 'hard']
        self.type = type

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
    
    def load_queries_v1(self, filename: str, query_type: str = ''):
        '''
        Loading queries from a pickle file with the CQD original format
        '''

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
        
    def load_queries_v2(self, folder_path: str, split: str):
        '''
        Loading queries from a pickle file with the Is complex query answering really complex? format
        '''
        filename = f'{folder_path}/{split}-queries.pkl'
        hardfile = f'{folder_path}/{split}-hard-answers.pkl'
        easyfile = f'{folder_path}/{split}-easy-answers.pkl'
        try:
            with open(filename, 'rb') as f:
                file_data = pickle.load(f)
                print("Loaded queries from", filename)
            with open(hardfile, 'rb') as f:
                hard_data = pickle.load(f)
                print("Loaded hard answers from", hardfile)
            with open(easyfile, 'rb') as f:
                easy_data = pickle.load(f)
                print("Loaded easy answers from", easyfile)

            for structure, queries in file_data.items():
                query_type = self.query_structure_to_type[structure]
                # skipping negation queries
                if 'n' in query_type:
                    continue
                for query in queries:
                    if self.type == 'complete':
                        # both hard and easy answers
                        hardanswers = list(hard_data.get(query, {}))
                        easyanswers = list(easy_data.get(query, {}))
                        final_answers = hardanswers + easyanswers
                    elif self.type == 'hard':
                        final_answers = list(hard_data.get(query, {}))
                    query = query_v2_to_v1(query, query_type)
                    self.add_query(query_type, (query, final_answers))
        except FileNotFoundError:
            raise ValueError(f'File {filename} not found')
        except Exception as e:
            raise ValueError(f'Error loading queries from {filename}: {e}')
        
