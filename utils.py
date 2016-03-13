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
