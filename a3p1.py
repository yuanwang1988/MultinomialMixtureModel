import numpy as np 
import scipy.io

np.set_printoptions(threshold=np.nan)

from utils import movie_filter
from MultinomialMM import expectation_step, max_step, rating_freq, log_likelihood, diagnosticK1

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


#################################
#Expectation Maximization
#################################

print '=========================='
print 'Expectation Maximization'
print '=========================='

print '------------------------'
print 'DiagnosticTesting K = 1'
print '------------------------'

diagnosticK1(train_movies, 5, 'hard')
diagnosticK1(train_movies, 5, 'soft')

print '------------------'
print 'Initialization:'
print '------------------'

#Setting Parameters
N = train_movies.shape[0]
M = train_movies.shape[1]
V = 5
K = 25

hard_itr = 5
soft_itr = 5

print 'Settings:'
print 'N: {}, M: {}, V: {}, K: {}'.format(N, M, V, K)
print 'Hard assignment - {} iterations'.format(hard_itr)
print 'Soft assignment - {} iterations'.format(soft_itr)

print '-------'

#Initialize beta and theta
theta = np.ones(K)/K
beta = np.random.uniform(1, 100, (K, M, V))
for k in xrange(K):
	for m in xrange(M):
		beta[k,m,:] = beta[k,m,:]/np.sum(beta[k,m,:])

rating_frequency = rating_freq(train_movies, V)

#print initial results
print 'Shape of theta: {}'.format(theta.shape)
print 'Shape of beta: {}'.format(beta.shape)

ll = log_likelihood(train_movies, beta, theta)

print 'Log likelihood: {}'.format(ll)

print '------------------'
print 'clustering:'
print '------------------'

for i in xrange(hard_itr):
	Q_z = expectation_step(train_movies, beta, theta)
	(beta, theta) = max_step(train_movies, Q_z, V)
	ll = log_likelihood(train_movies, beta, theta)
	print 'Log likelihood: {}'.format(ll)


beta_marg = np.zeros([M,V])
for k in xrange(K):
	beta_marg += beta[k,:,:]*theta[k]

print 'check: {}'.format(np.sum(np.square(beta_marg-rating_frequency)))


print '------------------'
print 'Multinomial Mix:'
print '------------------'
for i in xrange(soft_itr):
	Q_z = expectation_step(train_movies, beta, theta)
	(beta, theta) = max_step(train_movies, Q_z, V)
	ll = log_likelihood(train_movies, beta, theta)
	print 'Log likelihood: {}'.format(ll)


# print theta
# print beta


# print rating_freq

beta_marg = np.zeros([M,V])
for k in xrange(K):
	beta_marg += beta[k,:,:]*theta[k]

print 'check: {}'.format(np.sum(np.square(beta_marg-rating_frequency)))