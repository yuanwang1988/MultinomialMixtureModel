import numpy as np 
import scipy.io
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

from utils import movie_filter
from MultinomialMM import expectation_step, max_step, rating_freq, log_likelihood, diagnosticK1, MultinomialMixtureModel, expected_complete_data_loglikelihood

a3data_dic = scipy.io.loadmat('a3dataFinal.mat')

#print a3data_dic


print '=========================='
print 'Load Raw Data'
print '=========================='

print 'Data Fields:'

for key in a3data_dic:
	print key
	if key in set(['__header__', '__globals__', '__version__']):
		print a3data_dic[key]
	else:
		print a3data_dic[key].shape

print '---------'

print 'Raw training data size: {}'.format(a3data_dic['train_data'].shape)
print 'Raw test data size: {}'.format(a3data_dic['test_data'].shape)

# print 'Train data: example'
# print a3data_dic['train_user_indices'][0][0]
# print a3data_dic['train_data'][0]

# print 'Test data: example'
# print a3data_dic['test_user_indices'][0][0]
# print a3data_dic['test_data'][0]


################################
#Load Training and Testing Data
################################
print '=========================='
print 'Pre-processed Data'
print '=========================='

train_movies = a3data_dic['train_data']
filtered_movie_indices = movie_filter(train_movies, 200)
train_movies = train_movies[:, filtered_movie_indices]

test_movies = a3data_dic['test_data']
test_movies = test_movies[:, filtered_movie_indices]


print 'Training set size: {}'.format(train_movies.shape)

print 'Test set size: {}'.format(test_movies.shape)


########################################################
#Use cross-validation to find best number of clusters
########################################################
sofItr = 5
hardItr = 25

nFolds = 8
nRandStarts = 5

kf = KFold(train_movies.shape[0], nFolds, shuffle=True, random_state = None)

nClusterArray = [1, 2, 3, 4, 5, 8, 10, 15]
#nClusterArray = [1]
trainLLMat = np.zeros([len(nClusterArray), nFolds, nRandStarts])
validLLMat = np.zeros([len(nClusterArray), nFolds, nRandStarts])

for K_idx in xrange(len(nClusterArray)):
	K = nClusterArray[K_idx]

	fold_idx = 0
	for train, valid in kf:
		for s in xrange(nRandStarts):
			print '#############################################################'
			print 'nClusters = {}, fold = {}, randStart = {}'.format(K, fold_idx, s)
			print '#############################################################'
			MultinomialMM_model = MultinomialMixtureModel(5, K)
			MultinomialMM_model.fit(train_movies[train], sofItr, hardItr, 0.000000001)

			train_ll = MultinomialMM_model.eval(train_movies[train])
			valid_ll = MultinomialMM_model.eval(train_movies[valid])

			trainLLMat[K_idx, fold_idx, s] = train_ll
			validLLMat[K_idx, fold_idx, s] = valid_ll

			#save results
			np.savez('a3p1crossval3', nClusterArray = nClusterArray, trainLLMat = trainLLMat, validLLMat = validLLMat)

		fold_idx = fold_idx + 1


###################################################
#(A) Visualize cross-validation results
###################################################
Nvalid = train_movies.shape[0]/8
Ntrain = train_movies.shape[0] - Nvalid


a3p1crossval = np.load('a3p1crossval3.npz')

nClusterArray = a3p1crossval['nClusterArray']
trainLLMat = a3p1crossval['trainLLMat']
validLLMat = a3p1crossval['validLLMat']

#scale the log-likelihood by number of examples:
trainLLMat = trainLLMat/Ntrain
validLLMat = validLLMat/Nvalid

#average
trainLLAvg = np.average(np.average(trainLLMat, 2), 1)
validLLAvg = np.average(np.average(validLLMat, 2), 1)

#flatten
trainLLFlat = np.reshape(trainLLMat, [trainLLMat.shape[0], trainLLMat.shape[1]*trainLLMat.shape[2]])
validLLFlat = np.reshape(validLLMat, [validLLMat.shape[0], validLLMat.shape[1]*validLLMat.shape[2]])

print 'nClusters: {}'.format(nClusterArray)
print 'trainLLAvg: {}'.format(trainLLAvg)
print 'validLLAvg: {}'.format(validLLAvg)


#plot log-likelihood:
fig, ax = plt.subplots()
ax.plot(nClusterArray, trainLLFlat, 'b:')
ax.plot(nClusterArray, validLLFlat, 'r:')
ax.plot(nClusterArray, trainLLAvg, 'b', marker = 'o', markersize = 8, linewidth=3, label = 'Log-likelihood on training set (Avg)')
ax.plot(nClusterArray, validLLAvg, 'r', marker = 'o', markersize = 8, linewidth=3, label = 'Log-likelihood on validation set (Avg)')

#set title, axis labels and legend
ax.set_title('Log-likelihood of model on training and validation set (scaled) vs. number of clusters')
ax.set_xlabel('Number of clusters (K)')
ax.set_ylabel('Log-likelihood / Number of Examples')
legend = ax.legend(loc='lower left')

plt.show()

#plot log-likelihood:
fig, ax = plt.subplots()
width = 0.35
ind = np.arange(nClusterArray.shape[0])
rects1 = ax.bar(ind, (-1.0)*trainLLAvg, width, color='b')
rects2 = ax.bar(ind + width, (-1.0)*validLLAvg, width, color='r')

ax.set_xticks(ind+width)
ax.set_xticklabels(nClusterArray)

ax.set_title('Negative Log-likelihood of model on training and validation set (scaled) vs. number of clusters')
ax.set_xlabel('Number of clusters (K)')
ax.set_ylabel('Negative Log-likelihood / Number of Examples')
#legend = ax.legend(loc='upper right')
ax.legend((rects1[0], rects2[0]), ('Log-likelihood on training data', 'Log-likelihood on validation data'))

plt.show()


# #################################
# #Expectation Maximization
# #################################

# print '=========================='
# print 'Expectation Maximization'
# print '=========================='

# print '------------------------'
# print 'DiagnosticTesting K = 1'
# print '------------------------'

# diagnosticK1(train_movies, 5, 'hard')
# diagnosticK1(train_movies, 5, 'soft')

# print '------------------'
# print 'Initialization:'
# print '------------------'

# #Setting Parameters
# N = train_movies.shape[0]
# M = train_movies.shape[1]
# V = 5
# K = 2

# hard_itr = 5
# soft_itr = 5

# print 'Settings:'
# print 'N: {}, M: {}, V: {}, K: {}'.format(N, M, V, K)
# print 'Hard assignment - {} iterations'.format(hard_itr)
# print 'Soft assignment - {} iterations'.format(soft_itr)

# print '-------'

# #Initialize beta and theta
# theta = np.ones(K)/K
# beta = np.random.uniform(1, 100, (K, M, V))
# for k in xrange(K):
# 	for m in xrange(M):
# 		beta[k,m,:] = beta[k,m,:]/np.sum(beta[k,m,:])

# rating_frequency = rating_freq(train_movies, V)

# #print initial results
# print 'Shape of theta: {}'.format(theta.shape)
# print 'Shape of beta: {}'.format(beta.shape)

# ll = log_likelihood(train_movies, beta, theta)

# print 'Log likelihood: {}'.format(ll)

# print '------------------'
# print 'clustering:'
# print '------------------'

# for i in xrange(hard_itr):
# 	Q_z = expectation_step(train_movies, beta, theta)
# 	(beta, theta) = max_step(train_movies, Q_z, V)
# 	ll = log_likelihood(train_movies, beta, theta)
# 	print 'Log likelihood: {}'.format(ll)


# beta_marg = np.zeros([M,V])
# for k in xrange(K):
# 	beta_marg += beta[k,:,:]*theta[k]

# print 'check: {}'.format(np.sum(np.square(beta_marg-rating_frequency)))


# print '------------------'
# print 'Multinomial Mix:'
# print '------------------'
# for i in xrange(soft_itr):
# 	Q_z = expectation_step(train_movies, beta, theta)
# 	exp_comp_data_ll_E = expected_complete_data_loglikelihood(train_movies, Q_z, beta, theta)
# 	(beta, theta) = max_step(train_movies, Q_z, V)
# 	ll = log_likelihood(train_movies, beta, theta)
# 	exp_comp_data_ll_M = expected_complete_data_loglikelihood(train_movies, Q_z, beta, theta)
# 	print 'Log likelihood: {}'.format(ll)
# 	print 'Expected Complete Data Log-likelihood E-step: {}'.format(exp_comp_data_ll_E)
# 	print 'Expected Complete Data Log-likelihood M-step: {}'.format(exp_comp_data_ll_M)


# validLL = log_likelihood(test_movies, beta, theta)
# print 'Valid Log likelihood: {}'.format(validLL)

# # print theta
# # print beta


# # print rating_freq

# beta_marg = np.zeros([M,V])
# for k in xrange(K):
# 	beta_marg += beta[k,:,:]*theta[k]

# print 'check: {}'.format(np.sum(np.square(beta_marg-rating_frequency)))


#np.savez('Neill', beta = beta, theta = theta, trainLL = ll, validLL = validLL)