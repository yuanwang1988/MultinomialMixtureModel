import numpy as np 
import scipy.io
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

from utils import movie_filter, matrix2triplets, triplets2matrix
from MultinomialMM import expectation_step, max_step, rating_freq, log_likelihood, diagnosticK1, MultinomialMixtureModel, expected_complete_data_loglikelihood
from VariationalMatrixFactorization import prepare_data, initialize_param_estimate, variational_param_update, RMSEeval


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


print 'Check mat2triplet and triplet2mat:'
train_movies_triplets = matrix2triplets(train_movies)
train_movies_recon = triplets2matrix(train_movies_triplets, train_movies.shape[0], train_movies.shape[1])
print 'Sum Square Diff: {}'.format(np.sum(np.square(train_movies-train_movies_recon)))


#########################################
#2C Train Varational Inference Model
#########################################

print '==========================='
print 'Train Varational Model'
print '==========================='

N = train_movies.shape[0]
M = train_movies.shape[1]

#key settings
hyperparams = {'sigma':1.0, 'sigma_u':1.0, 'sigma_v':1.0}
K = 5
itrNum = 10
nFolds = 3

#k-fold
train_movies_triplets_complete = matrix2triplets(train_movies)
kf = KFold(train_movies_triplets_complete.shape[0], nFolds, shuffle=True, random_state = None)

for train, valid in kf:
	train_triplets = train_movies_triplets_complete[train]
	valid_triplets = train_movies_triplets_complete[valid]

	train_movies_mat = triplets2matrix(train_triplets, N, M)
	valid_movies_mat = triplets2matrix(valid_triplets, N, M)

	data = prepare_data(train_movies_mat)
	(Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma) = initialize_param_estimate(data, hyperparams, K)
	RMSE_train = RMSEeval(train_movies_mat, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams)
	RMSE_valid = RMSEeval(valid_movies_mat, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams)

	for itr in xrange(itrNum):
		(Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma) = variational_param_update(data, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams)
		RMSE_train = RMSEeval(train_movies_mat, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams)
		RMSE_valid = RMSEeval(valid_movies_mat, Q_u_mean, Q_u_sigma, Q_v_mean, Q_v_sigma, hyperparams)

		print 'RMSE train: {}; RMSE valid: {}'.format(RMSE_train, RMSE_valid)
