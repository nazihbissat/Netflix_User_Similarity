import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import time
from itertools import combinations
from itertools import product
import csv

# Set the random seed to 1 to get same results every run
np.random.seed(1)

# Function to load the data: computing shingle matrix and shingle dictionary for future use
def load_data():

    print('Loading data...')
    t = time.process_time()
    data_dict = dict()

    with open('netflix_data.txt', 'r') as netflix:
        movie_id = 0
        # Parse through data line by line
        for line in netflix:
            # If line contains MovieID, save MovieID
            if ':' in line:
                movie_id = int(line.split(':')[0])
            else:
                user_info = line.split(',')[0:2]
                user_id = user_info[0]
                rating = int(user_info[1])
                # Check if rating is greater than or equal to 3
                if rating >= 3:
                    # Insert UserID and information into data dictionary
                    if user_id in data_dict:
                        data_dict[user_id].add(movie_id)
                    else:
                        data_dict[user_id] = {movie_id}

    # Initialize list of columns
    data_columns = []
    movie_index = dict()
    movie_counter = 0

    for user_id in list(data_dict.keys()):
        # If user has watched less than 20 movies, delete information
        if len(data_dict[user_id]) > 20:
            del (data_dict[user_id])
        # Otherwise, enter information into data array
        else:
            user_entry = np.zeros(shape=(4507, 1))
            movies = data_dict[user_id]
            for movie_id in movies:
                if movie_id in movie_index:
                    user_entry[movie_index[movie_id]] = 1
                else:
                    movie_index[movie_id] = movie_counter
                    movie_counter += 1
                    user_entry[movie_counter] = 1
            data_columns.append(user_entry)

    # Concatenate all columns to form data matrix
    data_array = np.concatenate(data_columns, axis=1)

    # Create user index
    n = len(list(data_dict.keys()))
    user_index = dict(zip(range(n), list(data_dict.keys())))

    elapsed_time = time.process_time() - t
    print('Data loaded...')
    print('Time elapsed: ', str(elapsed_time) + ' s')

    return [data_array, data_dict, user_index, movie_index]


# Function to calculate the Jaccard distance between two sets a and b
def jaccard_dist(a, b):
    return 1 - len(a.intersection(b)) / len(a.union(b))


# Function to pick n random pairs from a matrix and compute their Jaccard similarity
def rand_jacc(n, data_dict):
    print('Randomly sampling 10,000 pairs of users and calculating Jaccard distances...')
    t = time.process_time()
    jacc_dists = []
    for i in np.arange(0, n, 1):
        pair = np.random.choice(list(data_dict.keys()), 2)
        jacc_dists.append(jaccard_dist(data_dict[pair[0]], data_dict[pair[1]]))
    elapsed_time = time.process_time() - t
    print('Jaccard distances calculated, histogram saved...')
    print('Time elapsed: ', str(elapsed_time) + ' s')
    print(jacc_dists)
    return jacc_dists


# Function to plot the histogram of Jaccard distances
def jaccard_histogram(jacc_dists):
    num_bins = 20
    n, bins, patches = plt.hist(jacc_dists, num_bins, facecolor='blue', alpha=0.5)
    plt.ylabel('Jaccard Distance')
    plt.title('Histogram of Jaccard Distances for 10,000 randomly sampled pairs')
    plt.savefig('histogram_q2')
    return [np.mean(np.array(jacc_dists)), np.min(jacc_dists)]


# Function to generate signature matrix from initial shingle matrix (numpy array), takes number of hashes (m) as input
def sig_matrix(shingle_dict, m, d=4507):

    print('Computing signature matrix with 1000 hash functions...')
    t = time.process_time()
    columns = list(shingle_dict.keys())
    n = len(columns)

    # Initialize signature matrix to infinity in all positions
    final_matrix = np.ones(shape=(m, n)) * np.inf

    # Generate m random hash coefficients and store them in a dictionary
    hash_a = np.random.choice((np.arange(1, d, 1)), m)
    hash_b = np.random.choice((np.arange(1, d, 1)), m)

    # For every column (every key in data dictionary)
    for i in range(len(columns)):
        user_id = user_index[i]
        # For every row containing a 1
        movies = list(shingle_dict[user_id])
        # Calculate hash of row for every hash function
        for movie_id in movies:
            perms = np.remainder(hash_a * movie_index[movie_id] + hash_b, d)
            final_matrix[:, i] = np.minimum(perms, final_matrix[:, i])

    np.save('m_hash_a', hash_a)
    np.save('m_hash_b', hash_b)
    np.save('signature_matrix.npy', final_matrix)
    elapsed_time = time.process_time() - t
    print('Signature matrix created...')
    print('Time elapsed: ', str(elapsed_time) + ' s')

    return [final_matrix, [hash_a, hash_b]]


# Function that returns all pairs of similar users (prime is a large prime number, set to 10,000th prime number)
def all_similar_users(sig_matrix, r, b, prime=104729):

    print('Locating pairs of similar users...')
    t = time.process_time()
    n = len(sig_matrix[0,:])

    # Initialize list of bands
    bands = []
    for i in np.arange(0, b, 1):
        band = sig_matrix[i * r : (i + 1) * r, :]
        bands.append(band)

    # Generate r pairs hash coefficients
    hash_a = np.random.choice(list(np.arange(1, prime, 1)), r)
    hash_b = np.random.choice(list(np.arange(1, prime, 1)), r)

    # Initialize set of tuples of similar users
    all_pairs = set()

    reduced_sig_rows = []
    all_buckets = []

    # For every band
    for b in bands:
        row = []
        buckets = dict()
        # For every column (every key in data dictionary)
        for i in range(n):
            user_id = user_index[i]
            # For every row containing a 1
            v = b[:, i]
            # Calculate hash value of each row using its hash function and sum them up
            bucket = np.sum(np.remainder(np.multiply(hash_a,v) + hash_b, prime))
            # Append hash value to row of reduced signature matrix
            row.append(bucket)
            # If hash value exists in dictionary of buckets
            if bucket in buckets:
                # Compute every combination of current user with all user in bucket and add to set of candidate pairs
                for p in list(product(list(buckets[bucket]), [user_id])):
                    if jaccard_dist(data_dict[p[0]], data_dict[p[1]]) <= 0.35:
                        all_pairs.add(p)
                buckets[bucket].add(user_id)
            # Otherwise, add user to new bucket
            else:
                buckets[bucket] = {user_id}
        # Append row of reduced signature matrix
        reduced_sig_rows.append(row)
        # Append list of buckets for band b to list of all buckets
        all_buckets.append(buckets)

    # Concatenate all rows of reduced signature matrix
    reduced_sig_matrix = np.concatenate(reduced_sig_rows, axis=0)

    np.save('reduced_sig_matrix', reduced_sig_matrix)
    np.save('r_hash_a', hash_a)
    np.save('r_hash_b', hash_b)

    print('Pairs of similar users located...')
    elapsed_time = time.process_time() - t
    print('Time elapsed: ', str(elapsed_time) + ' s')

    return [all_pairs, [hash_a, hash_b], reduced_sig_matrix, all_buckets]


# Function that takes a new user vector, the signature matrix, and the shingle dictionary as input and outputs the most
# similar user to the one with the input ID
def nearest_neighbor(new_user, all_buckets, m_hash_a, m_hash_b, r_hash_a, r_hash_b, movie_index,
                  data_dict, m=1000, d=4507, r=10, b=100, prime=104729):

    print('Locating approximate nearest neighbour of queried user...')

    # Create a set containing the movies that the queried user liked
    liked_movies = set()
    for row in new_user:
        if row == 1:
            liked_movies.add(row)

    # Compute the queried user's column in the signature matrix
    sig_col = np.ones(shape=(m,)) * np.inf
    for movie in liked_movies:
        perms = np.remainder(m_hash_a * movie_index[movie] + m_hash_b, d)
        sig_col = np.minimum(perms, sig_col[:, ])

    # Partition the column into b bands
    c_neighbors = set()
    bands = []
    for i in np.arange(0, b, 1):
        band = sig_col[i*r:(i+1)*r,]
        bands.append(band)

    for i in range(len(bands)):
        band = bands[i]
        bucket = np.sum(np.remainder(np.multiply(r_hash_a, band) + r_hash_b, prime))
        # If there is a user(s) in this bucket already, add the ID(s) to the set of candidate pairs
        if bucket in all_buckets[i]:
            c_neighbors = all_buckets[i][bucket]|c_neighbors

    # Now that we have a list of candidate neighbors, find the one with the minimal Jaccard distance to queried user
    neighbor = [np.inf, 0]
    for user_id in c_neighbors:
        dist = jaccard_dist(liked_movies, data_dict[user_id])
        if dist < neighbor[0]:
            neighbor = [dist, user_id]

    if neighbor == [np.inf, 0]:
      return 'There are no users within a Jaccard distance of 0.35 from the queried user...'
    else:
        return neighbor[1]


# Script to run code above

# # Question 1:
info = load_data()
data_matrix = info[0]
data_dict = info[1]
user_index = info[2]
movie_index = info[3]

# # Question 2:
stats = jaccard_histogram(rand_jacc(10000, data_dict))
mean = stats[0]
min = stats[1]
print('The minimum Jaccard distance is: ' + str(min) + '...')

# # Question 3:
sig_matrix_output = sig_matrix(data_dict, 1000, d=4507)
sig_matrix = sig_matrix_output[0]
m_hash_a = sig_matrix_output[1][0]
m_hash_b = sig_matrix_output[1][1]

# # Question 4:
similar_user_output = all_similar_users(sig_matrix,10,100)
similar_user_pairs = similar_user_output[0]
r_hash_a = similar_user_output[1][0]
r_hash_b = similar_user_output[1][1]
reduced_sig_matrix = similar_user_output[2]
all_buckets = similar_user_output[3]

print('There are ' + str(len(similar_user_pairs)) + ' pairs of similar users...')

# with open('similarPairs.csv','w') as writeFile:
#   similarWriter = csv.writer(writeFile, delimiter=',')
#   for pair in similar_user_pairs:
#     similarWriter.writerow([pair[0], pair[1]])

i = 0
distances = []
for pair in similar_user_pairs:
    distances.append(jaccard_dist(data_dict[pair[0]], data_dict[pair[1]]))
    if i > 10000:
        break
    i += 1
print('The mean Jaccard distance of 10,000 randomly sampled similar pairs: ' + str(np.mean(np.array(distances))))
print('The mean Jaccard distance of 10,000 randomly sampled pairs: ' + str(mean))

# Question 5
new_user = []
nearest_neighbor(new_user, all_buckets, m_hash_a, m_hash_b, r_hash_a, r_hash_b, movie_index, data_dict)