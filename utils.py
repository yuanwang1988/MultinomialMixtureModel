import numpy as np

def movie_filter(rating_matrix, vote_threshold):
	'''
	Purpose: remove movies that do have have enough votes (num of votes < threshold

	Input:
		- rating matrix - NxM array where each row represent a user and each column represent a movie
		- vote threshold - integer representing minimum number of votes for a movie to be considered

	Output:
		- array_of_indices - array of indices of movies that have >= vote_thresold number of votes
	'''

	array_of_indices = []

	for j in xrange(rating_matrix.shape[1]):
		if np.count_nonzero(rating_matrix[:, j]) >= vote_threshold:
			array_of_indices.append(j)

	return array_of_indices

def matrix2triplets(rating_matrix):
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	# print 'mat2triplet N: {}'.format(N)
	# print 'mat2triplet M: {}'.format(M)


	user_movie_rating_triplets = np.zeros([np.count_nonzero(rating_matrix), 3])

	triplet_idx = 0
	for i in xrange(N):
		for j in xrange(M):
			if rating_matrix[i,j]>0:
				user_movie_rating_triplets[triplet_idx,:] = [i,j,rating_matrix[i,j]]
				triplet_idx = triplet_idx + 1

	return user_movie_rating_triplets

def triplets2matrix(user_movie_rating_triplets, nUsers, nMovies):
	N = nUsers
	M = nMovies
	# print 'triplet2mat N: {}'.format(N)
	# print 'triplet2mat M: {}'.format(M)

	rating_matrix = np.zeros([N,M])

	for t in xrange(user_movie_rating_triplets.shape[0]):
		user_idx = user_movie_rating_triplets[t,0]
		movie_idx = user_movie_rating_triplets[t,1]
		rating = user_movie_rating_triplets[t,2]
		rating_matrix[user_idx, movie_idx] = rating

	return rating_matrix


