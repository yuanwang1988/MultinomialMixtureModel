import numpy as np
from scipy.sparse import dok_matrix


def prepare_data(rating_matrix):

	'''
	Initialize the data structures for holding data relevant
	to the Variational Matrix factorization algorithm.
	'''

	data = {}

	#get dimensions:
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	
	#transform rating matrix into a sparse matrix
	sparse_rating_matrix = dok_matrix((N,M), dtype=np.float32)
	for i in xrange(rating_matrix.shape[0]):
		for j in xrange(rating_matrix.shape[1]):
			if rating_matrix[i,j]>0:
				sparse_rating_matrix[i,j] = rating_matrix[i,j]

	#construct user_neighb_dic and movie_neighb_dic
	#user_neighb_dic maps users -> movies he/she rated
	user_neighb_dic = {}
	for i in xrange(rating_matrix.shape[0]):
		user_neighb_dic[i] = np.nonzero(rating_matrix[i,:])[0]
	
	#movie_neighb_dic maps movies -> users that rated the move
	movie_neighb_dic = {}
	for j in xrange(rating_matrix.shape[1]):
		movie_neighb_dic[j] = np.nonzero(rating_matrix[:,j])[0]


	data['N'] = N
	data['M'] = M
	data['ratings'] = sparse_rating_matrix
	data['user_neighb_dic'] = user_neighb_dic
	data['movie_neighb_dic'] = movie_neighb_dic

	return data


def initialize_param_estimate(data, hyperparams, K):
	'''
	Initialize the parameter estimates.

	 ratings - dok_matrix((N,M), dtype=np.float32)
	 N - number of users
	 M - number of movies
	 K - number of factors
	 user_neighb_dic - {key: index of user, value: list of indices of movies}
	 movie_neighb_dic - {key: index of movies, value: list of indices of user}

	 Q_u_mean - N x K
	 Q_u_sigma - N x K x K
	 Q_v_mean - M x K
	 Q_v_sigma - M x K x K

	'''

	#sigma, sigma_u, sigma_v

	#Obtain data:
	N = data['N']
	M = data['M']
	ratings = data['ratings']
	user_neighb_dic = data['user_neighb_dic']
	movie_neighb_dic = data['movie_neighb_dic']

	#Obtain hyperparameters
	sigma = hyperparams['sigma']
	sigma_u = hyperparams['sigma_u']
	sigma_v = hyperparams['sigma_v']

	##################
	#intialize params
	##################
	Q_u_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K)*sigma_u, N)
	Q_u_sigma = np.ones([N,K,K])*sigma_u

	Q_v_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K)*sigma_v, M)
	Q_v_sigma = np.ones([M,K,K])*sigma_v


	return (Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma)


def variational_param_update(data, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams):
	'''
	Perform one iteration of variational update on the model parameter estimate.

	 ratings - dok_matrix((N,M), dtype=np.float32)
	 N - number of users
	 M - number of movies
	 K - number of factors
	 user_neighb_dic - {key: index of user, value: list of indices of movies rated}
	 movie_neighb_dic - {key: index of movies, value: list of indices of user that rated movie} 

	 Q_u_mean - N x K
	 Q_u_sigma - N x K x K
	 Q_v_mean - M x K
	 Q_v_sigma - M x K x K
	'''

	#sigma, sigma_u, sigma_v

	#Obtain data:
	N = data['N']
	M = data['M']
	K = Q_u_mean.shape[1]
	ratings = data['ratings']
	user_neighb_dic = data['user_neighb_dic']
	movie_neighb_dic = data['movie_neighb_dic']

	#Obtain hyperparameters
	sigma = hyperparams['sigma']
	sigma_u = hyperparams['sigma_u']
	sigma_v = hyperparams['sigma_v']

	########################
	#Initialize S_j and t_j
	########################

	#S_tensor[j-1,:,:] is S_j in the paper
	S_tensor = np.zeros([M, K, K])
	for j in xrange(M):
		S_tensor[j,:,:] = np.identity(K)*(1/np.square(sigma_v))

	#t_array[j-1] is the t_j in the paper
	t_array = np.zeros([M,K])

	#iterate through all of the users
	for i in xrange(N):
		#Update Q_u_sigma, Q_u_mean for user i
		
		###################
		#compute Q_u_sigma
		###################
		#compute the sum of squares of the v_mean_j
		neighbs_of_i = user_neighb_dic[i]
		sum_v_meanjv_meanjT = np.zeros([K,K])
		for j in neighbs_of_i:
			sum_v_meanjv_meanjT = sum_v_meanjv_meanjT + np.outer(Q_v_mean[j,:], Q_v_mean[j,:])

		Q_u_sigma_i_inv = np.identity(K)*(1/np.square(sigma_u)) + np.square(1/sigma)*(np.sum(Q_v_sigma[neighbs_of_i], 0) + sum_v_meanjv_meanjT)
		Q_u_sigma_i = np.linalg.inv(Q_u_sigma_i_inv)
		
		###################
		#compute Q_u_mean
		###################
		#compute sum_{over j}rij*v_mean_j
		neighbs_of_i = user_neighb_dic[i]
		sum_rij_v_meanj = np.zeros(K)

		for j in neighbs_of_i:
			sum_rij_v_meanj = sum_rij_v_meanj+ratings[i,j]*Q_v_mean[j,:]

		Q_u_mean_i = np.square(1/sigma)*np.matmul(Q_u_sigma_i, sum_rij_v_meanj)

		############################
		#store Q_u_mean, Q_u_sigma
		############################
		Q_u_mean[i,:] = Q_u_mean_i
		Q_u_sigma[i,:,:] = Q_u_sigma_i

		#####################################################
		#Update Sj and tj for all j in neighbourhood of i
		#####################################################
		neighbs_of_i = user_neighb_dic[i]
		for j in neighbs_of_i:
			S_tensor[j,:,:] = S_tensor[j,:,:] + np.square(1/sigma)*(Q_u_sigma_i + np.outer(Q_u_mean_i, Q_u_mean_i))
			t_array[j,:] = t_array[j,:] + np.square(1/sigma)*(ratings[i,j]*Q_u_mean_i.T)

	######################################
	#Update Q_v_mean, Q_v_sigma
	######################################
	for j in xrange(M):
		Q_v_sigma[j,:,:] = np.linalg.inv(S_tensor[j,:,:])

		temp =  np.matmul(Q_v_sigma[j,:,:], t_array[j,:])

		Q_v_mean[j,:] = temp

	return (Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma)


def RMSEeval(test_ratings, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams):

	'''
	Compute the RMSE between the predicted rating
	and actual ratings

	'''

	pred_out = np.matmul(Q_u_mean, Q_v_mean.T)
	sqr_diff = np.square(test_ratings - pred_out)

	sqr_diff_masked = np.ma.masked_array(sqr_diff, mask = test_ratings==0)
	RMSE = np.sqrt(np.mean(sqr_diff_masked))

	return RMSE