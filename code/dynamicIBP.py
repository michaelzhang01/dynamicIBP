# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:57:32 2016

Dynamic IBP code, some parts taken from Ke Zhai's IBP code

@author: mz6268
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from past.builtins import xrange
from past.builtins import reduce
#import sys
import time
import numpy
import pickle
import matplotlib.pyplot as plt
import datetime
import os
from copy import copy
from scipy import stats
#from scipy.misc import logsumexp
from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
numpy.set_printoptions(precision=3)


def gendata_symbols(isIBP=False, sX=0.1):
    A = numpy.array([[1,1,1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],      [0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0],      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0]])
    N=200
    #sX=0.1
    alpha=1.
    lamb=0.5
    L = numpy.zeros((N,4))
    B = numpy.zeros((N,4))
    Z = (numpy.random.uniform(0,1,(N,4))<0.5).astype(numpy.int)
    L = numpy.random.geometric(lamb, size=(N,4))
    L=L*Z
    if isIBP is True:
        L=numpy.copy(Z)
    B=copy(Z)
    Y = numpy.zeros((N,4))
    for row_index, L_n in enumerate(L):
        max_lifetime = L_n.max() #longest lifetime in that row.
        lifetime_Y = numpy.zeros((max_lifetime, 4))
        for L_index,L_val in enumerate(L_n):
            #                pdb.set_trace()
            lifetime_Y[:L_val, L_index] = B[row_index, L_index]
        if row_index + max_lifetime < N:
            Y[row_index:row_index+max_lifetime,:] += lifetime_Y
        else:
            Y[row_index:,:] += lifetime_Y[:(N-row_index),:]
    if sX==0:
        X=numpy.dot(Y,A)
    else:
        X = numpy.dot(Y,A) +numpy.random.normal(loc=0, scale=sX, size=(N,36))
    return Z,L,B,Y,A,X

def factors(n):
    return numpy.sort(list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(numpy.sqrt(n)) + 1) if n % i == 0)))))

class dynamicIBP(object):
    def __init__(self, data, alpha=1., lmbda = None, sigma_a=1., sigma_x=1.,
                 initial_Y = None, initial_K = None, initial_L=None, 
                 initial_B = None, initial_A = None, 
                 alpha_hyper_parameter=(.1,1.0),
                 sigma_a_hyper_parameter=(1.0,1.0),
                 sigma_x_hyper_parameter=(1.0,1.0), 
                 B_hyper_parameter=(1.0,1.0),
                 lambda_hyper_parameter=(1.0,1.0), B_prior = 'flat',
                 isIBP = False, mask=None, collapsed=True,
                 output_folder = "../output",  test_frac = .1,
                 filename ="tmp", A_lik = "gauss"):
        """
        Init parameters
    
        data -- NxD numpy array,
        alpha -- alpha parameter, float
        lmbda -- lambda parameter, float
        sigma_a -- A standard deviation, float
        sigma_x -- X standard deviation, float
        initial_Y -- initialization for duration of feature, npy array
        initial_K -- intial number of features, int
        initial_L -- initialization for geometric draws, numpy array
        initial_A -- initialization for features, numpy array
        initial_B -- initialization for feature weights, numpy array
        alpha_hyper_parameter -- hyper parameters for alpha, 2 dim tuple
        sigma_a_hyper_parameter -- hyper parameters for sigma_a, 2 dim tuple
        sigma_x_hyper_parameter -- hyper parameters for sigma_a, 2 dim tuple
        lambda_hyper_parameter=(1.0,1.0) -- hyper parameters for lambda
        B_prior -- choice of prior on B, either 'normal', 'flat' or 'gamma'
        B_hyper_parameter -- hyper parameters for B (ampltidue) for normal it 
            is (mean, standard deviation), for gamma it is (shape, scale), 
            2 dim tuple
        A_lik -- likelihood of feature matrix, either 'gauss' or 'laplace'
        isIBP -- if True, run static IBP otherwise run dynamic IBP
        mask -- binary matrix where 0 represents observed data and 1 represents
                missing data / held out test set data
        collapsed -- if True, sample with A integrated out
        output_folder -- directory to write output results
        test_frac -- percentage of observations to hold out if mask is None
        filename -- filename for outputfiles
        
        Initialize object with dynamicIBP() and run sampler with method 
        dynamicIBP.sample()    
    
        Test data is the last test_frac percent of the data, arranged in 
        checkerboard pattern
        """

        self.A_scale = 1.
        self._it = 0
        self._filename = filename+".p"
        self._llname = filename+"_ll.txt"
        self._msename = filename+"+mse.txt"
        self._isIBP = isIBP
        self._sigma_a = sigma_a
        self._sigma_x = sigma_x
        self._alpha = alpha
        self._test_frac = test_frac
        assert((self._test_frac < 1.) and (self._test_frac > 0.))
        self.today = datetime.datetime.today()
        self._alpha_hyper_parameter = alpha_hyper_parameter
        self._sigma_a_hyper_parameter = sigma_a_hyper_parameter
        self._sigma_x_hyper_parameter = sigma_x_hyper_parameter
        self._B_hyper_parameter = B_hyper_parameter
        self._B_prior = B_prior # only 'flat', 'gamma', or 'normal'
        assert((self._B_prior == 'flat') or (self._B_prior == 'gamma') or (self._B_prior == 'normal'))
        self._B_method = 'random_walk' # this is always random walk?
        self._lambda_hyper_parameter = lambda_hyper_parameter
        self._X = data
        (self._N, self._D) = self._X.shape
        self._A_lik = A_lik
        assert((self._A_lik == 'gauss') or (self._A_lik == 'laplace'))
        if self._A_lik == 'laplace':
            self._isCollapsed = False
        else:
            self._isCollapsed = collapsed
        self._output_folder = output_folder
        
        if mask is None:
            if self._D % 2 == 0:
                even_mask = tuple([0,1]*(self._D//2))
                odd_mask = tuple([1,0]*(self._D//2))
            else:
                even_mask = [0,1]*(self._D//2 + 1)
                even_mask = tuple(even_mask[:self._D])
                odd_mask = [1,0]*(self._D//2 + 1 )
                odd_mask = tuple(odd_mask[:self._D])
            idx_mask = numpy.array([even_mask,odd_mask] * (int(self._N * self._test_frac )/ int(2)))
            idx_N = idx_mask.shape[0]
            zeros_mask = numpy.zeros((self._N - idx_N, self._D))
            self.mask = numpy.vstack((zeros_mask,idx_mask)).astype(int)

        else:
            self.mask = mask

        self.test_idx = self.mask.sum(axis=1).nonzero()[0]
        self.X_test = numpy.copy(self._X)

        # initialization code
        if initial_K is None:
            self._K = sum([numpy.random.poisson(self._alpha / (n+1.)) for n in xrange(self._N)])
        else:
            self._K = initial_K

        assert(self._K > 0)

        if lmbda is  None:
            self.lmbda = numpy.random.beta(self._lambda_hyper_parameter[0],
                                           self._lambda_hyper_parameter[1],
                                           size=self._K)
        else:
            self.lmbda_val = lmbda
            assert(lmbda >= 0 and lmbda <= 1)
            self.lmbda = numpy.array([lmbda]*self._K)


        if initial_L is None:
            Z = numpy.zeros((self._N, self._K))
            self._L = copy(Z)
        else:
            self.initial_L = initial_L
            self._L = initial_L#numpy.minimum(initial_L,1)
            assert(self._L.shape == (self._N, self._K))
        self._L = self._L.astype(int)


        if initial_B is None:
            if self._B_prior == 'flat':
                self._B = numpy.minimum(self._L,1)
            else:
                self._B = numpy.random.gamma(self._B_hyper_parameter[0],
                                             self._B_hyper_parameter[1],
                                             size=(self._N, self._K))
        else:
            self.initial_B = initial_B
            self._B = initial_B
            
        self._B = self._B*(self._L>0) #explicitly making non-zero parameters zero here.

        if initial_Y is None:
            self._Y = self.LB_to_Y(self._L, self._B) #Y is now the full feature loaing matrix
        else:
            self._Y = initial_Y
            self.initial_Y = initial_Y
            assert(self._Y.shape == (self._N, self._K))

        if A_lik == "gauss":
            self._M = self.compute_M(self._Y)
            sgn, self.log_det_M = numpy.linalg.slogdet(self._M)

        if initial_A != None:
            self._A = initial_A;
        else:
            self._A = self.init_map_estimate_A();
        assert(self._A.shape == (self._K, self._D));

    def save(self):
        f = open(self._filename, "wb")
        pickle.dump(self.__dict__,f)
        f.close()
        self.saveText()

    def load(self):
        f = open(self._filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def LB_to_Y(self, L, B):
        N,K = L.shape
        Y = numpy.zeros(L.shape)
        for row_index, L_n in enumerate(L):
            L_n = L_n.astype(int)
            max_lifetime = L_n.max() #longest lifetime in that row.
            lifetime_Y = numpy.zeros((max_lifetime, K))
            for L_index,L_val in enumerate(L_n):
                L_val = int(L_val)
                lifetime_Y[:L_val, L_index] = B[row_index, L_index]
            if row_index + max_lifetime < N:
                Y[row_index:row_index+max_lifetime,:] += lifetime_Y
            else:
                Y[row_index:,:] += lifetime_Y[:(N-row_index),:]
        return(Y)

    def update_Y(self, n,k, newL=None, newB=None):
        if newL is None:
            newL = self._L[n,k]
        if newB is None:
            newB = self._B[n,k]
        Y = numpy.copy(self._Y)
        
        #remove old
        if self._L[n,k]+n>self._N:
            Y[n:,k] -= self._B[n,k]
        else:
            Y[n:(n+self._L[n,k]),k] -= self._B[n,k]
        if n+newL > self._N:
            Y[n:,k]+=newB
        else:
            Y[n:(n+newL),k]+=newB
        return Y

    def init_map_estimate_A(self):
        (mean, std_dev) = self.init_sufficient_statistics_A();
        assert(mean.shape == (self._K, self._D));
        return mean

    def compute_M(self, Y=None):
        if Y is None:
            Y = self._Y;

        K = Y.shape[1];
        M = numpy.linalg.inv(numpy.dot(Y.T, Y) + (self._sigma_x / self._sigma_a) ** 2 * numpy.eye(K));
        return M

    def init_sufficient_statistics_A(self):
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = self.compute_M(self._Y);
        # compute the mean of the matrix A
        mean_A = numpy.dot(M, numpy.dot(self._Y.T, self._X));
        # compute the co-variance of the matrix A
        std_dev_A = numpy.linalg.cholesky(self._sigma_x ** 2 * M).T;

        return (mean_A, std_dev_A)

    def uncollapsed_log_likelihood_X(self, Y=None,X=None,A=None):
        if Y is None:
            Y=self._Y
        if X is None:
            X=self._X
        if A is None:
            A=self._A
        (N, D) = X.shape
        Xtilde = numpy.dot(Y,A)
        diff = X-Xtilde
        log_likelihood = -N * D * numpy.log(2 * numpy.pi) -numpy.sum((diff**2)/(2*self._sigma_x**2))
        return log_likelihood

    def collapsed_log_likelihood_X(self, Y=None, X=None,
                                   M = None, log_det_M=None):
        if Y is None:
            Y = self._Y;
        if X is None:
            X = self._X
        if M is None:
            M = self._M
        if log_det_M is None:
            log_det_M = self.log_det_M

        assert(X.shape[0] == Y.shape[0]);
        (N, D) = X.shape;
        (N, K) = Y.shape;
        assert(M.shape == (K, K));
        log_likelihood = numpy.eye(N) - numpy.dot(numpy.dot(Y, M), Y.transpose());
        log_likelihood = -0.5 / (self._sigma_x ** 2) * numpy.trace(numpy.dot(numpy.dot(X.T, log_likelihood), X));
        log_likelihood -= D * (N - K) * numpy.log(self._sigma_x) + K * D * numpy.log(self._sigma_a);
        log_likelihood += 0.5 * D * log_det_M;
        log_likelihood -= 0.5 * N * D * numpy.log(2 * numpy.pi);
        return(log_likelihood)

    def sample_missing(self):
        for n in self.test_idx:
            ya_n = numpy.dot(self._Y[n,:],self._A) + self._sigma_x*numpy.random.normal(loc=0,scale=self._sigma_x,size = self._D)
            self._X[n,numpy.where(self.mask[n,:]==1)] = ya_n[numpy.where(self.mask[n,:]==1)]


    def sample_Bn(self, L_ind, M_i, log_det_M_i):
        MH_count = numpy.zeros(2)
        for K_ind in xrange(self._K):
            if self._L[L_ind, K_ind]>0: #no point sampling if it's always zero'd out
                new_B = numpy.copy(self._B)
                if self._isCollapsed:
                    current_prob = self.collapsed_log_likelihood_X()
                else:
                    current_prob = self.uncollapsed_log_likelihood_X()
                if self._B_prior == 'gamma':
                    if self._B_method == 'random_walk':
                        stepsize = 0.01
                        new_B[L_ind, K_ind] = -1
                        while new_B[L_ind, K_ind]<0:
                            new_B[L_ind, K_ind] =self._B[L_ind, K_ind] +stepsize*numpy.random.randn()
                    else:
                        D_draw = numpy.random.gamma(self._B_hyper_parameter[0], self._B_hyper_parameter[1])
                        new_B[L_ind, K_ind] = D_draw
                elif self._B_prior == 'normal':
                    D_draw = numpy.random.normal(self._B_hyper_parameter[0], self._B_hyper_parameter[1])
                    new_B[L_ind, K_ind] = D_draw
                new_Y = self.update_Y(L_ind, K_ind, newB= new_B[L_ind, K_ind])
                if self._A_lik == "gauss":
                    if self._isCollapsed:
                        new_M = self.compute_M(new_Y)
                        sgn,new_log_det_M = numpy.linalg.slogdet(new_M)
                if self._isCollapsed:
                    new_prob = self.collapsed_log_likelihood_X(Y = new_Y, M=new_M, log_det_M=new_log_det_M)
                    if self._B_method =='random_walk':
                        new_prob += stats.gamma.logpdf(new_B[L_ind, K_ind], self._B_hyper_parameter[0],scale=self._B_hyper_parameter[1])
                        current_prob +=stats.gamma.logpdf(self._B[L_ind, K_ind], self._B_hyper_parameter[0],scale=self._B_hyper_parameter[1])
                    MH_prob = numpy.exp(new_prob - current_prob)
                    MH_count[1] += 1
                    if numpy.random.uniform() < MH_prob:
                        MH_count[0] += 1
                        self._B = new_B
                        self._M = new_M
                        self._Y = new_Y
                        self.log_det_M = new_log_det_M
                else:  #if uncollapsed
                    lifetime = self._L[L_ind, K_ind]
                    if L_ind+lifetime<self._N:
                        Xtilde_current = numpy.dot(self._Y[L_ind:(L_ind+lifetime),:],self._A)
                        diff_current = self._X[L_ind:(L_ind+lifetime),:]-Xtilde_current
                        Xtilde_new = numpy.dot(new_Y[L_ind:(L_ind+lifetime),:],self._A)
                        diff_new =self._X[L_ind:(L_ind+lifetime),:]-Xtilde_new
                    else:
                        Xtilde_current = numpy.dot(self._Y[L_ind:self._N,:],self._A)
                        diff_current = self._X[L_ind:self._N,:]-Xtilde_current
                        Xtilde_new = numpy.dot(new_Y[L_ind:self._N,:],self._A)
                        diff_new = self._X[L_ind:self._N,:]-Xtilde_new
                    current_prob= -numpy.sum((diff_current**2)/(2*self._sigma_x**2))
                    new_prob = -numpy.sum((diff_new**2)/(2*self._sigma_x**2))
                    if self._B_method =='random_walk':
                        new_prob += stats.gamma.logpdf(new_B[L_ind, K_ind], self._B_hyper_parameter[0],scale=self._B_hyper_parameter[1])
                        current_prob +=stats.gamma.logpdf(self._B[L_ind, K_ind], self._B_hyper_parameter[0],scale=self._B_hyper_parameter[1])
                    MH_prob= numpy.exp(new_prob - current_prob)
                    MH_count[1]+=1
                    if numpy.random.uniform() < MH_prob:
                        MH_count[0] += 1
                        self._B = new_B
                        self._Y = new_Y
        return(MH_count)


    def sample_L_slice(self, L_ind, M_i=None, log_det_M_i=None):
        accept_count = numpy.zeros(2)
        Z = (self._L > 0).astype(int)
        m_k = (Z.sum(axis=0) - Z[L_ind,:]).astype(float)
        log_prob_z1 = numpy.log(m_k + self._alpha/self._N)
        log_prob_z0 = numpy.log(self._N - m_k + self._alpha/self._N)
        for K_ind in xrange(self._K):
            u = numpy.log(numpy.random.rand())
            L_nk = self._L[L_ind, K_ind]
            assert(L_nk >= 0)
            if self._isIBP:
                bracket = [0,2]
            else:
                bb = 10
                bracket = [numpy.maximum(0, L_nk-bb), numpy.minimum(self._N-L_ind+1, L_nk+bb)]

            num_slice = 0
            orig_bracket = copy(bracket)
            accept_L = False

            if L_nk == 0:
                current_prior = log_prob_z0[K_ind]
            else:
                current_prior = log_prob_z1[K_ind]
                if self._isIBP is False:
                    current_prior = current_prior+ L_nk*numpy.log(1-self.lmbda[K_ind]) + numpy.log(self.lmbda[K_ind])

            if self._isCollapsed is True:
                tmp_M = numpy.copy(self._M)
                self._M = self.compute_M()
                sgn, self.log_det_M= numpy.linalg.slogdet(self._M)
                if numpy.any(tmp_M-self._M):
                    raise ValueError
                current_prob = self.collapsed_log_likelihood_X() + current_prior
            else:
                current_diff = self._X - numpy.dot(self._Y,self._A)
                current_prob = -numpy.sum((current_diff**2)/(2*self._sigma_x**2)) + current_prior

            accept_threshold = u+current_prob
            L_proposal = numpy.copy(self._L)
            B_proposal = numpy.copy(self._B)
            if self._L[L_ind, K_ind]==0:
                if self._B_prior == "flat":
                    B_proposal[L_ind, K_ind]=1
                else:
                    B_proposal[L_ind, K_ind] = numpy.random.gamma(self._B_hyper_parameter[0], self._B_hyper_parameter[1])
            while accept_L is False:
                num_slice+=1
                proposal_L_nk = bracket[0] + numpy.random.randint(bracket[1]-bracket[0])
                if self._isIBP:
                    if proposal_L_nk>1:
                        raise ValueError
                L_proposal[L_ind, K_ind] = proposal_L_nk
                if proposal_L_nk==0:
                    proposal_prior = log_prob_z0[K_ind]
                else:
                    proposal_prior = log_prob_z1[K_ind]
                    if self._isIBP is False:
                        proposal_prior = proposal_prior+ proposal_L_nk*numpy.log(1-self.lmbda[K_ind])+numpy.log(self.lmbda[K_ind])
                Y_proposal = self.update_Y(L_ind,K_ind,newL = L_proposal[L_ind, K_ind], newB = B_proposal[L_ind, K_ind])#proposal_L_nk, newB= Bprop)
                if self._isCollapsed:
                    M_proposal = self.compute_M(Y_proposal)#self.compute_M(Y_proposal*self._B)
                    sgn, log_det_M_proposal = numpy.linalg.slogdet(M_proposal)
                    proposal_prob = self.collapsed_log_likelihood_X(Y=Y_proposal, M = M_proposal, log_det_M = log_det_M_proposal) + proposal_prior
                else:
                    new_diff = self._X - numpy.dot(Y_proposal,self._A)
                    proposal_prob = -numpy.sum((new_diff**2)/(2*self._sigma_x**2)) + proposal_prior

                if proposal_prob>accept_threshold:
                    self._L = L_proposal
                    self._B = B_proposal
                    if self._L[L_ind, K_ind]==0:
                        self._B[L_ind, K_ind] = 0
                    self._Y = Y_proposal
                    if self._isCollapsed:
                        self._M = M_proposal
                        self.log_det_M = log_det_M_proposal
                    accept_L = True

                else:
                    if self._isIBP is True:
                        accept_L = True
                    else:
                        if proposal_L_nk==L_nk:
                            raise ValueError
                        if proposal_L_nk>L_nk:
                            bracket[1]=proposal_L_nk
                        else:
                            bracket[0]=proposal_L_nk
                        if bracket[1]<=L_nk:
                            raise ValueError
                        if num_slice>self._N:
                            raise ValueError

    def sample_sigma(self, sigma_hyper_parameter, matrix):
        assert(sigma_hyper_parameter != None);
        assert(~numpy.any(matrix==None)); #make sure no values are none
        assert(type(sigma_hyper_parameter) == tuple);
        (sigma_hyper_a, sigma_hyper_b) = sigma_hyper_parameter;
        (row, column) = matrix.shape;
        posterior_shape = sigma_hyper_a + 0.5 * row * column;
        var = 0;
        if row >= column:
            var = numpy.trace(numpy.dot(matrix.transpose(), matrix));
        else:
            var = numpy.trace(numpy.dot(matrix, matrix.transpose()));

        posterior_scale = 1.0 / (sigma_hyper_b + var * 0.5);
        if posterior_scale<=0:
            raise ValueError
        tau = numpy.random.gamma(posterior_shape, posterior_scale);
        sigma_a_new = numpy.sqrt(1.0 / tau);
        return sigma_a_new;

    def sample_alpha(self):
        assert(self._alpha_hyper_parameter != None);
        assert(type(self._alpha_hyper_parameter) == tuple);
        (alpha_hyper_a, alpha_hyper_b) = self._alpha_hyper_parameter;
        posterior_shape = alpha_hyper_a + self._K;
        H_N = numpy.array([range(self._N)]) + 1.0;
        H_N = numpy.sum(1.0 / H_N);
        posterior_scale = 1.0 / (alpha_hyper_b + H_N);
        alpha_new = numpy.random.gamma(posterior_shape, posterior_scale);
        return alpha_new;

    def regularize_matrices(self):
        Z = (self._L >0).astype(int)
        assert(self._Y.shape == (self._N, self._K));
        Z_sum = numpy.sum(Z, axis=0)
        assert(len(Z_sum) == self._K);
        indices = numpy.nonzero(Z_sum == 0);
        keep_idx = [k for k in range(self._K) if k not in indices[0]]
        self._L = self._L[:,keep_idx]
        self._Y = self._Y[:,keep_idx]
        self._B = self._B[:,keep_idx]
        self.lmbda = self.lmbda[keep_idx]
        assert(self._Y.shape == self._L.shape)
        self._K = self._Y.shape[1];
        assert(self._L.shape == (self._N, self._K));
        assert(self._Y.shape == (self._N, self._K));
        assert(self._B.shape == (self._N, self._K));
        assert(self.lmbda.size == self._K)
        if self._isCollapsed:
            self._M = self.compute_M(self._Y)
            sgn, self.log_det_M = numpy.linalg.slogdet(self._M)
        if self._isCollapsed is False:
            self._A = self._A[keep_idx,:]

    def compute_M_i(self, L_ind, Y=None, B=None, M=None, log_det_M=None):
        if Y is None:
            Y = self._Y
        if B is None:
            B = self._B
        if M is None:
            M = self._M
        if log_det_M is None:
            log_det_M = self.log_det_M

        MyiyiM = numpy.dot(numpy.dot(M,numpy.dot(Y[L_ind,:].T,Y[L_ind,:])),M)
        yiMyi = numpy.dot(numpy.dot(Y[L_ind,:],M),Y[L_ind,:].T)
        M_i = M - (MyiyiM / (yiMyi - 1.));
        log_det_M_i = log_det_M - numpy.log(1. - yiMyi);
        return(M_i, log_det_M_i)

    def saveText(self):
        likelihood_txt_fn = os.path.abspath(self._llname)
        mse_txt_fn = os.path.abspath(self._msename)
        numpy.savetxt(likelihood_txt_fn, self.likelihood_iteration, delimiter=",", header="time, log likelihood")
        numpy.savetxt(mse_txt_fn, self.MSE_iteration, delimiter=",", header="time, MSE")

    def plotStuff(self):
        if int(numpy.sqrt(self._D))**2 == self._D:
            factor_1 = factor_2 = numpy.sqrt(self._D)
        else:
            D_factors = factors(self._D)
            factor_len = len(D_factors)
            factor_1 = D_factors[factor_len//2]
            factor_2 = self._D // factor_1
        assert(factor_1 * factor_2 == int(self._D))
        feat_plot, feat_axes = plt.subplots(1, self._K, figsize=(10,2))
        for k_i, k_v in enumerate(xrange(self._K)):
            if factor_1 > factor_2:
                feat_axes[k_v].imshow(self._A[k_v].reshape(factor_1, factor_2), interpolation="none", cmap='gist_gray')
            else:
                feat_axes[k_v].imshow(self._A[k_v].reshape(factor_2, factor_1), interpolation="none", cmap='gist_gray')
            feat_axes[k_v].set_title(str(k_v))
            feat_axes[k_v].axis('off')

        feat_plot.subplots_adjust(top = .8, hspace=0, wspace= .05)
        abs_min = numpy.min((self._X,numpy.dot(self._Y*self._B,self._A)))
        abs_max = numpy.max((self._X,numpy.dot(self._Y*self._B,self._A)))
        tick_loc = numpy.linspace(abs_min,abs_max,5)
        f, axes = plt.subplots(2,1, figsize=(10,8), sharex=True, sharey='all')
        im0=axes[0].imshow(self._X.T, aspect='auto', interpolation='None', cmap='bwr', vmin=abs_min, vmax=abs_max)
        axes[0].set_title("True Signal", fontsize=12)
        axes[0].set_ylabel("Dimension")
        div = make_axes_locatable(axes[0])
        cax = div.append_axes("right", size="3%", pad=0.1)
        f.colorbar(im0, cax=cax, ticks=tick_loc, format="%.1f")

        im1=axes[1].imshow(numpy.dot(self._Y, self._A).T, aspect='auto', interpolation='None', cmap='bwr', vmin=abs_min, vmax=abs_max)
        axes[1].set_title("Reconstructed Signal", fontsize=12)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Dimension")
        div = make_axes_locatable(axes[1])
        cax = div.append_axes("right", size="3%", pad=0.1)
        f.colorbar(im1, cax=cax, ticks=tick_loc, format="%.1f")
        f.tight_layout()
        f.subplots_adjust(hspace=.1, wspace=0)

        life_plot, life_ax = plt.subplots(figsize=(11.65,4))
        life_im = life_ax.imshow((self._Y).T, aspect='auto', interpolation='None')
        life_ax.set_ylabel("Feature")
        life_ax.set_xlabel("Time")
        life_div = make_axes_locatable(life_ax)
        life_cax = life_div.append_axes("right", size="3%", pad=0.1)
        life_ax.yaxis.set_ticks(xrange(self._K))
        life_ax.set_title("Feature Intensity", fontsize=12)
        life_plot.colorbar(life_im, cax=life_cax,  format="%.1f")
        LL_plot, LL_ax = plt.subplots(figsize=(11.65,4))
        LL_ax.plot(numpy.log(self.likelihood_iteration[:,0]), self.likelihood_iteration[:,1])
        LL_ax.set_xlabel("Log Time (s)")
        LL_ax.set_ylabel("Log Likelihood")
        LL_plot.suptitle("Log Likelihood vs. Time", fontsize=12)

        MSE_plot, MSE_ax = plt.subplots(figsize=(11.65,4))
        MSE_ax.plot(numpy.log(self.MSE_iteration[:,0]), self.MSE_iteration[:,1])
        MSE_ax.set_xlabel("Log Time (s)")
        MSE_ax.set_ylabel("Mean Squared Error")
        MSE_plot.suptitle("MSE vs. Time", fontsize=12)
        if self._isIBP is True:
            image_foot = "_IBP_slice" + self.today.strftime("%Y-%m-%d-%f") + ".png"
        else:
            image_foot = "_dIBP_slice" + self.today.strftime("%Y-%m-%d-%f") + ".png"

        feat_int_fn = os.path.abspath(self._output_folder + "feautre_intensity" + image_foot)
        data_fn = os.path.abspath(self._output_folder + "data" + image_foot)
        features_fn = os.path.abspath(self._output_folder + "features" + image_foot)
        likelihood_plot_fn = os.path.abspath(self._output_folder + "log_likelihood" + image_foot)
        mse_plot_fn = os.path.abspath(self._output_folder + "mse" + image_foot)

        life_plot.savefig(feat_int_fn, dpi=600, format='png', bbox_inches='tight')
        feat_plot.savefig(features_fn, dpi=600, format='png', bbox_inches='tight')
        f.savefig(data_fn, dpi=600, format='png', bbox_inches='tight')
        LL_plot.savefig(likelihood_plot_fn, dpi=600, format='png', bbox_inches='tight')
        MSE_plot.savefig(mse_plot_fn, dpi=600, format='png', bbox_inches='tight')

    def sample_A(self):
        if self._A_lik == "gauss":
            self._M = self.compute_M(self._Y)
            self.log_det_M = None #test to see if there's any problem here
            mean_a = numpy.dot(self._M, numpy.dot(self._Y.T, self._X))
            chol_a = numpy.linalg.cholesky(self._sigma_x ** 2 * self._M)
            self._A = mean_a + numpy.dot(chol_a, numpy.random.normal(size=(self._K, self._D)))
            self._sigma_a = self.sample_sigma(self._sigma_a_hyper_parameter, self._A)
        elif self._A_lik == "laplace":
            for k in range(self._K):
                if numpy.any(self._Y[:,k]>0):
                    rel_n = numpy.where(self._Y[:,k]>0)
                    for d in range(self._D):
                        old_a = numpy.copy(self._A[k,d])
                        old_logprior = - numpy.abs(old_a)/self.A_scale
                        old_diff = self._X[rel_n,d] - numpy.dot(self._Y[rel_n,:], self._A[:,d])
                        old_loglik = -numpy.sum(old_diff**2)/(2*self._sigma_x**2)
                        old_lp = old_logprior + old_loglik
                        logu = numpy.log(numpy.random.rand())
                        thresh = old_lp + logu
                        window = 1.
                        interval = [old_a-window, old_a + window]
                        accepted = False
                        counter= 0
                        while not accepted:
                            self._A[k,d] = interval[0] + (interval[1]-interval[0])*numpy.random.rand()
                            prop_logprior = -numpy.abs(self._A[k,d])/self.A_scale
                            prop_diff = self._X[rel_n,d] - numpy.dot(self._Y[rel_n,:], self._A[:,d])
                            prop_loglik = -numpy.sum(prop_diff**2)/(2*self._sigma_x**2)
                            prop_lp = prop_logprior + prop_loglik
                            if prop_lp > thresh:
                                accepted = True
                            else:
                                if self._A[k,d]>old_a:
                                    interval[1] = numpy.copy(self._A[k,d])
                                else:
                                    interval[0] = numpy.copy(self._A[k,d])
                                if counter > 100:
                                    self._A[k,d] = numpy.copy(old_a)
                                    accepted=True
                                counter +=1
                else:
                    self._A[k,:] = numpy.random.laplace(loc=0, scale=self.A_scale, size = (1,self._D))

        else:
            raise ValueError
            
    def sample(self, iters = 200, saveEvery = 5):
        """
        Sample parameters    
        iters -- iterations, int
        saveEvery -- save results at every "saveEvery" iteration
        """
        self.likelihood_iteration = numpy.zeros((iters, 2))
        self.MSE_iteration = numpy.zeros((iters, 2))
        start_time = time.time()
        while self._it<iters:
            for L_ind in xrange(self._N):
                self.sample_L_slice(L_ind)#M_i, log_det_M_i, stop_samp = False)
            if self._B_prior == 'flat':
                B_MH = 1
            else:
                B_MH = self.sample_Bn(L_ind, M_i=None, log_det_M_i=None)            
            self._Y = self.LB_to_Y(self._L, self._B)
            lmbda_beta_a = self._lambda_hyper_parameter[0] + (self._L>0).sum(axis=0)
            lmbda_beta_b = self._lambda_hyper_parameter[1] + self._L.sum(axis=0)-(self._L>0).sum(axis=0)
            self.lmbda = numpy.random.beta(lmbda_beta_a, lmbda_beta_b)
            assert(self.lmbda.size == self._K)
            self.sample_A()
            self.sample_missing()
            self._sigma_x = self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.dot(self._Y, self._A));
            elapsed = time.time()-start_time
            self._M = self.compute_M()
            sgn, self.log_det_M= numpy.linalg.slogdet(self._M)
            YA = numpy.dot(self._Y, self._A)
            MSE_test = (self.X_test[self.mask==1] - YA[self.mask==1])**2 #self.X_test is the raw X
            MSE_train = (self.X_test[self.mask==0]-YA[self.mask==0])**2
            MSE = MSE_test.mean()
            MSE_train = MSE_train.mean()
            self.MSE_iteration[self._it,:] = [self._it, MSE]

            if self._isCollapsed:
                print("Collapsed, iter: %i\tK: %i\tMSE: %.2f\ttrain MSE: %.2f" % (self._it, self._K,  MSE, MSE_train));
            else:
                print("Uncollapsed, iter: %i\tK: %i\tMSE: %.2f\ttrain MSE: %.2f" % (self._it, self._K,  MSE, MSE_train));
            print("alpha: %.2f\tsigma_a: %.2f\tsigma_x: %.2f\ttime: %.2f" % (self._alpha, self._sigma_a, self._sigma_x, elapsed));
            Z = self._L > 0
            print("lambda: %s\tfeat. count: %s" % (self.lmbda, Z.sum(axis=0)))
            if self._it%saveEvery==0:
                self.save()
            self._it+=1
            if self._it == iters:
                self.save()


if __name__ == "__main__":
    bird_data = loadmat('../data/bird_sounds.mat')
    X = bird_data['XX']
    mask = bird_data['Xmask']
    filename = "../output/bird_sounds_dIBP"+str(0)+".p"        
    dibp= dynamicIBP(data=X, lmbda = 0.5, initial_K=20,isIBP = False,
                     mask = mask, B_prior="normal", collapsed=True, 
                     filename=filename)
    dibp.sample(iters=1000)        
