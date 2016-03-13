import numpy as np 
np.set_printoptions(threshold=np.nan)

################################
#EM Functions - Public
################################

def expectation_step(rating_matrix, old_beta, old_theta):
	'''
	Purpose: take the rating_matrix (observed), the old_beta and old_theta and 
	return the p(Z|R, old_beta, old_theta)

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- old_beta - KxMxV - rows are the clusters, columns are the movies, the 3rd axis are probability of each rating (usually 1 to 5)
		- old_theta - Kx1 - probablity of each cluster

	Output:
		- Q_z - NxK - p(Z|R, old_beta, old_theta) - posterior probability of z being in each one of the clusters. 
		Each row is a user, each column is a cluster
	'''
	#get dimensions:
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	V = old_beta.shape[2]
	K = old_theta.shape[0]

	#Compute the unnormalized Q_z in log space:
	log_Q_z = unnorm_log_Q_z(rating_matrix, old_beta, old_theta)
	Q_z = np.zeros([N, K])

	#Normalize the Q_z (i.e. so each row sums to 1)
	log_Q_z_rowsums = np.log(np.sum(np.exp(log_Q_z), 1))
	#print log_Q_z_rowsums

	for n in xrange(N):
		Q_z[n,:] = np.exp(log_Q_z[n,:] - log_Q_z_rowsums[n])

	#print Q_z.shape

	#print Q_z


	return Q_z

def hard_assign(rating_matrix, old_beta, old_theta):
	'''
	Purpose: take the rating_matrix (observed), the old_beta and old_theta and 
	return a hard assignment of each user to a cluster

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- old_beta - KxMxV - rows are the clusters, columns are the movies, the 3rd axis are probability of each rating (usually 1 to 5)
		- old_theta - Kx1 - probablity of each cluster

	Output:
		- Q_hard - NxK - hard assign each user to a cluster that maixmizesp(Z|R, old_beta, old_theta) 
		posterior probability of z being in each one of the clusters. 
		Each row is a user, each column is a cluster
	'''
	#get dimensions:
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	V = old_beta.shape[2]
	K = old_theta.shape[0]

	log_Q_z = unnorm_log_Q_z(rating_matrix, old_beta, old_theta)

	Q_hard = np.zeros([N, K])

	for n in xrange(N):
		k_max = np.argmax(unnorm_log_Q_z[n,:])
		Q_hard[n, k_max] = 1

	return Q_hard

def max_step(rating_matrix, Q_z, V):
	'''
	Purpose: take the rating_matrix (observed) and Q_z=p(Z|R, old_beta, old_theta),
	return the updated estimate for beta and theta

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- Q_z - NxK - p(Z|R, old_beta, old_theta) - posterior probability of z being in each one of the clusters. 
		Each row is a user, each column is a cluster
		- V - integer indicating the maximum rating

	Output:
		- new_beta - KxMxV - rows are the clusters, columns are the movies, the 3rd axis are probability of each rating (usually 1 to 5)
		- new_theta - Kx1 - probablity of each cluster
	'''
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	V = V
	K = Q_z.shape[1]

	#get machine epsilon
	eps = np.finfo(np.float).eps


	#transform rating matrix to 1-hot - NxMxV 
	rating_matrix_transformed = rating_matrix_1hot_transform(rating_matrix, V)

	#compute the new beta and theta
	new_theta = (np.sum(Q_z, 0)/N).T

	new_beta = np.zeros([K, M, V])
	new_beta_count = np.zeros([K, M, V])

	#get unormalized beta -> i.e. the counts weighted by responsibility
	for k in xrange(K):
		for m in xrange(M):
			for v in xrange(V):
				new_beta_count[k, m, v] = np.dot(Q_z[:,k], rating_matrix_transformed[:,m,v])

	#compute noramlized new_beta
	#sum along the V axis (3rd axis, axis = 2)
	new_beta_count = new_beta_count + eps
	new_beta_count_sums = np.sum(new_beta_count, 2)

	for k in xrange(K):
		for m in xrange(M):
			new_beta[k,m,:] = new_beta_count[k,m,:]/new_beta_count_sums[k,m]


	#print new_theta
	#print np.sum(new_theta)
	#print np.sum(new_beta, 2)

	return (new_beta, new_theta)

##########################################
#Likelihood Function - Public
##########################################


def log_likelihood(rating_matrix, beta, theta):
	'''
	Purpose: take the rating_matrix (observed), beta and theta and return the likelihood of the data

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		Each row is a user, each column is a cluster
		- beta - KxMxV - rows are the clusters, columns are the movies, the 3rd axis are probability of each rating (usually 1 to 5)
		- theta - Kx1 - probablity of each cluster

	Output:
		- log_likelihood - float indicating the log likelihood of the data
	'''

	#get dimensions:
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	V = beta.shape[2]
	K = theta.shape[0]

	log_Q_z = unnorm_log_Q_z(rating_matrix, beta, theta)
	log_likelihood = np.sum(np.log(np.sum(np.exp(log_Q_z), 1)))

	return log_likelihood


#################################
#Helper Functions:
#################################

def unnorm_log_Q_z(rating_matrix, old_beta, old_theta):
	'''
	Purpose: take the rating_matrix (observed), the old_beta and old_theta and 
	return the unnormalized log{p(Z|R, old_beta, old_theta)}

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- old_beta - KxMxV - rows are the clusters, columns are the movies, the 3rd axis are probability of each rating (usually 1 to 5)
		- old_theta - Kx1 - probablity of each cluster

	Output:
		- log_Q_z - NxK - unnormalized log_p(Z|R, old_beta, old_theta) - posterior probability of z being in each one of the clusters. 
		Each row is a user, each column is a cluster
	'''
	
	#get dimensions:
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]
	V = old_beta.shape[2]
	K = old_theta.shape[0]

	#get machine epsilon
	eps = np.finfo(np.float).eps

	old_beta = old_beta+eps
	old_theta = old_theta+eps

	#transform rating matrix to 1-hot:
	rating_matrix_transformed = rating_matrix_1hot_transform(rating_matrix, V)

	#Compute the unnormalized Q_z in log space:
	log_Q_z = np.zeros([N, K])

	#compute log_beta and log_theta
	log_old_beta = np.log(old_beta)
	log_old_theta = np.log(old_theta)

	for n in xrange(N):
		for k in xrange(K):
			temp_sum = 0
			for j in xrange(M):
				temp_sum += np.dot(rating_matrix_transformed[n, j, :], log_old_beta[k, j, :])

			# print temp_sum.shape
			# print log_old_theta[k]

			log_Q_z[n,k] = temp_sum + log_old_theta[k]


	return log_Q_z



##############################
#Utility Functions:
##############################
def rating_matrix_1hot_transform(rating_matrix, V):
	'''
	Purpose: take the rating_matrix (observed) and
	return the 1-hot version of the rating matrix

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- V - integer indicating the maximum rating

	Output:
		- rating_matrix_transformed - NxMxV - 1hot representation of movie ratings. The third axis is for ratings.
	'''
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]

	#convert the rating matrix to NxMxV - consider optimizing this if the E-step is slow
	rating_matrix_transformed = np.zeros([N, M, V])
	for i in xrange(N):
		for j in xrange(M):
			v = rating_matrix[i,j]
			if v > 0:
				rating_matrix_transformed[i, j, v-1] = 1

	return rating_matrix_transformed


########################################
#Reasonability Test Functions - Public
########################################
def rating_freq(rating_matrix, V):
	'''
	Purpose: take the rating_matrix (observed) and
	return the 1-hot version of the rating matrix

	Input:
		- rating_matrix - NxM - rows are users and columns are movies
		- V - integer indicating the maximum rating

	Output:
		- rating_freq: MxV - rows are movies, columns are ratings
	'''
	N = rating_matrix.shape[0]
	M = rating_matrix.shape[1]

	rating_matrix_transformed = rating_matrix_1hot_transform(rating_matrix, V)

	#for each movie and each rating, count the number of occurances
	rating_count = np.zeros([M, V])

	for m in xrange(M):
		for v in xrange(V):
			rating_count[m,v]=np.sum(rating_matrix_transformed[:,m,v])

	#for each movie and each rating, compute the frequency of ratings
	rating_freqs = np.zeros([M,V])

	for m in xrange(M):
		rating_freqs [m,:] = rating_count[m,:]/np.sum(rating_count[m,:])

	#check rows add up to 1

	#get machine epsilon
	eps = np.finfo(np.float).eps

	assert(np.sum(np.square(np.sum(rating_freqs, 1) - np.ones([M, 1])))<eps)

	return rating_freqs

def diagnosticK1(train_movies, V, hard_soft_flag):
	'''
	Purpose: perform diagnostic by comparing result of K=1 vs. simple frequency count. Expect to be the same.
	Input:
		- train_movies - NxM - rows are users and columns are movies
		- V - integer indicating the maximum rating
		- hard_soft_flag - string = {'hard', 'soft'} - indicate whether we want to test hard or soft assignment

	Output:
		- None
		- If test passed, print message; Otherwise, assertion fail
	'''
	N = train_movies.shape[0]
	M = train_movies.shape[1]

	V = 5
	K = 1

	hard_itr = 5
	soft_itr = 5

	#Initialize beta and theta
	theta = np.ones(K)/K
	beta = np.random.uniform(5, 10, (K, M, V))
	for k in xrange(K):
		for m in xrange(M):
			beta[k,m,:] = beta[k,m,:]/np.sum(beta[k,m,:])

	rating_frequency = rating_freq(train_movies, V)

	#print initial results
	# print 'Shape of theta: {}'.format(theta.shape)
	# print 'Shape of beta: {}'.format(beta.shape)

	ll = log_likelihood(train_movies, beta, theta)

	# print 'Log likelihood: {}'.format(ll)

	# print '------------------'
	# print 'clustering:'
	# print '------------------'
	if hard_soft_flag == 'hard':
		for i in xrange(hard_itr):
			Q_z = expectation_step(train_movies, beta, theta)
			(beta, theta) = max_step(train_movies, Q_z, V)
			ll = log_likelihood(train_movies, beta, theta)
			# print 'Log likelihood: {}'.format(ll)

		beta_marg = np.zeros([M,V])
		for k in xrange(K):
			beta_marg += beta[k,:,:]*theta[k]


	# print '------------------'
	# print 'Multinomial Mix:'
	# print '------------------'
	else:
		for i in xrange(soft_itr):
			Q_z = expectation_step(train_movies, beta, theta)
			(beta, theta) = max_step(train_movies, Q_z, V)
			ll = log_likelihood(train_movies, beta, theta)
			# print 'Log likelihood: {}'.format(ll)

		beta_marg = np.zeros([M,V])
		for k in xrange(K):
			beta_marg += beta[k,:,:]*theta[k]

	#check assertions:
	#get machine epsilon
	eps = np.finfo(np.float).eps

	#theta = 1
	assert(np.sum(np.square(theta - 1))<eps)

	#k-1 beta should be same as simple count (within machine precision)
	assert(np.sum(np.square(beta_marg-rating_frequency))<eps)

	print 'Reasonability Check Passed for: {} assignment'.format(hard_soft_flag)