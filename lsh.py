# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import numpy as np # for creating matrices or arrays
import random # for randomly generating a and b for hash functions
from itertools import combinations # for creating candidate pairs in lsh

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    k = parameters_dictionary["k"]
    docs_k_shingles = []  # holds the k-shingles of each document
    for ID, text in document_list.items():
        current_document = []
        word_list = text.split()
        for i in range(len(word_list)-k+1):
            current_shingle = word_list[i:i+k]
            current_document.append(" ".join(current_shingle))

        docs_k_shingles.append(current_document)

    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def k_shingles():
    k = parameters_dictionary["k"]
    docs_k_shingles = []  # holds the k-shingles of each document
    for ID, text in document_list.items():
        current_document = []
        word_list = text.split()
        for i in range(len(word_list)-k+1):
            current_shingle = word_list[i:i+k]
            current_document.append(" ".join(current_shingle))

        docs_k_shingles.append(current_document)

    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    docs_sig_sets = []
    

    unique_shingles=[]
    for shingle_list in k_shingles:
        unique_shingles.extend(shingle_list)
    
    shingle_dict={}
    unique_shingles=list(sorted(set(unique_shingles)))
    for i in range(len(unique_shingles)):
        shingle_dict[str(unique_shingles[i])]=i

    input_matrix = np.zeros([len(unique_shingles), len(k_shingles)],dtype=bool)
    
    for j, shingle_list in enumerate(k_shingles):
        for shingle in shingle_list:
            i=shingle_dict[shingle]
            input_matrix[i][j]=1
    
    docs_sig_sets=input_matrix.copy()
    return docs_sig_sets


def larger_prime(N):
    primes=[2]
    i=3
    while primes[-1]<=N:
        i_is_prime=True
        for p in primes:
            if i%p==0:
                i_is_prime=False
        if i_is_prime:
            primes.append(i)
        i+=1
    return primes[-1]

def gen_func(num_perm,p,N):
    a=random.randint(1,num_perm)
    b=random.randint(1,num_perm)
    return lambda r: ((a*r+b)%p)%N

# METHOD FOR TASK 3

# A function for generating hash functions
def generate_hash_functions(num_perm, N):
    hash_funcs = []
    p=larger_prime(N)
    
    for i in range(num_perm):
        x=gen_func(num_perm,p,N)
        while x in hash_funcs:
            x=gen_func(num_perm,p,N)
        hash_funcs.append(x)
    return hash_funcs

# Creates the minHash signatures after generating hash functions
def minHash(docs_signature_sets, hash_fn):
    min_hash_signatures = []

    num_rows, num_cols = docs_signature_sets.shape
    min_hash_signatures = np.full((len(hash_fn), num_cols), np.inf)

    for r in range(num_rows):
        for c in range(num_cols):
            if docs_signature_sets[r][c]:
                for i in range(len(hash_fn)):
                    hash_val = hash_fn[i](r)
                    if hash_val < min_hash_signatures[i][c]:
                        min_hash_signatures[i][c] = hash_val
    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    candidates = []  # list of candidate sets of documents for checking similarity

    

    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_dict = []

    # implement your code here

    return similarity_dict



# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    print(f"First 3 document shingles: {all_docs_k_shingles[0]}, {all_docs_k_shingles[1]},{all_docs_k_shingles[2]}")
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    hash_fn = generate_hash_functions(parameters_dictionary['permutations'], len(signature_sets))
    min_hash_signatures = minHash(signature_sets, hash_fn)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t'], "% similarity...")
    t14 = time.time()
    true_pairs = candidates_similarities(candidate_docs, min_hash_signatures)
    t15 = time.time()
    print(f"The total number of candidate pairs from LSH: {len(candidate_docs)}")
    print(f"The total number of true pairs from LSH: {len(true_pairs)}")
    print(f"The total number of false positives from LSH: {len(candidate_docs) - len(true_pairs)}")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t14 - t15, "sec")

    
    print("The pairs of documents are:\n")
    for p in true_pairs:
        print(f"LSH algorith reveals that the BBC article {list(p.keys())[0][0]+1}.txt and {list(p.keys())[0][1]+1}.txt \
              are {round(list(p.values())[0],2)*100}% similar")
        
        print("\n")

    
    
