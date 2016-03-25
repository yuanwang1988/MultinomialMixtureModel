import numpy as np

def variational_param_update(data, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma):
	# ratings - dok_matrix((N,M), dtype=np.float32)
	# N - number of users
	# M - number of movies
	# K - number of factors
	# user_neighb_dic - {key: index of user, value: list of indices of movies}
	# movie_neighb_dic - {key: index of movies, value: list of indices of user}

	#Q_u_mean - N x K
	#Q_u_sigma - N x K x K
	#Q_v_mean - M x K
	#Q_v_sigma - M x K x K

	#sigma, sigma_u, sigma_v

	#Initialize S_j and t_j

	#S_tensor[j-1,:,:] is S_j in the paper
	S_tensor = np.zeros([M, K, K])
	for j in xrange(M):
		S[j,:,:] = np.identity(K)*(1/np.square(sigma_v))

	#t_array[j-1] is the t_j in the paper
	t_array = np.zeros([M,K])

	#iterate through all of the users
	for i in xrange(N)
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
		sum_rij_v_meanj = np.zeros(M)
		for j in neighbs_of_i:
			sum_rij_v_meanj = ratings[i,j]*Q_v_mean[j,:]

		Q_u_mean_i = np.square(1/sigma)*np.matmul(Q_u_sigma_i, sum_rij_v_meanj.T).T

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
			t_array[j,:] = t_array[j,:] + np.square(1/sigma)*(ratings[i,j]*Q_u_mean_i)

	######################################
	#Update Q_v_mean, Q_v_sigma
	######################################
	for j in xrange(M):
		Q_v_sigma[j,:,:] = np.linalg.inv(S_tensor[j,:,:])
		Q_v_mean[j,:] = np.matmul(Q_v_sigma[j,:,:], t_array[j,:].T).T
