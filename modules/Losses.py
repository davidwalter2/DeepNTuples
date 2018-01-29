from keras import backend as K

from tensorflow import where, greater, abs, zeros_like, exp
import tensorflow as tf

global_loss_list={}

#whenever a new loss function is created, please add it to the global_loss_list dictionary!

def bce_weighted(y_true, y_pred):
    '''binary crossentropy, including weights
       y_true = [isSignal, eventweight]
       y_pred = [prob_isSignal]
    '''
    isSignal =      y_true[:,0]
    isSignalPred =  y_pred[:,0]
    weight = y_true[:,-1]
    return K.sum(weight * K.binary_crossentropy(isSignalPred,isSignal))/K.sum(weight)

global_loss_list['bce_weighted']=bce_weighted


def cce_weighted(y_true, y_pred):
    '''categorical crossentropy, including weights
       y_true = [isSignal_1, isSignla_2, ..., eventweight]
       y_pred = [prob_isSignal_1, prob_isSignal_2, ...]
    '''
    isSignal =      y_true[:,:-1]
    isSignalPred =  y_pred
    weight = y_true[:,-1]
    return K.sum(weight * K.categorical_crossentropy(isSignalPred,isSignal))/K.sum(weight)

global_loss_list['cce_weighted']=cce_weighted


def bce_weighted_dex(y_true, y_pred):
    '''binary crossentropy, including weights, data excluded
        for shuffled data/mc sample which only takes mc into account
       y_true = [isSignal,  isData, eventweight]
       y_pred = [prob_isSignal]
    '''
    isSignal = y_true[:,0] + y_true[:,1] + y_true[:,2]
    isSignalPred = y_pred[:,0] + y_pred[:,1] + y_pred[:,2]
    isData = y_true[:, -2:-1]
    weight = y_true[:, -1:]
    return K.sum(weight * K.binary_crossentropy(isSignalPred, isSignal) * (1 - isData)) / K.sum((1 - isData) * weight)

global_loss_list['bce_weighted']=bce_weighted_dex


def cce_weighted_dex(y_true, y_pred):
    '''binary crossentropy, including weights, data excluded
        for shuffled data/mc sample which only takes mc into account
       y_true = [isSignal_1, isSignla_2, ..., isData, eventweight]
       y_pred = [prob_isSignal_1, prob_isSignal_2, ...]
    '''
    isSignal = y_true[:,:-2]
    isSignalPred = y_pred
    isData = y_true[:, -2]
    weight = y_true[:, -1]
    return K.sum(weight * K.categorical_crossentropy(isSignalPred, isSignal) * (1 - isData)) / K.sum((1 - isData) * weight)

global_loss_list['cce_weighted_dex']=cce_weighted_dex


def moments_weighted(y_true, y_pred, momentum_weights = [1.,1.]):
    '''punishment linear to the difference of the means and variance of the prediction distributions from data and mc
       y_true = [isData, eventweight]
       y_pred = [prob_isSignal_1, prob_isSignal_2, ...]
       momentum_weights = [mean_weight, variance_weight]
    '''
    isData = y_true[:,:1]
    weight = y_true[:,1:2]
    for i in range(1,y_pred.shape[1]):  #for some reasons K.repeat doesn't work ...
        weight = K.concatenate((weight,y_true[:,1:2]), axis = 1)

    #computes the mean and variance value
    moments_data = tf.nn.weighted_moments(y_pred, axes = 0, frequency_weights = weight*isData)
    moments_mc = tf.nn.weighted_moments(y_pred, axes = 0, frequency_weights = weight*(1-isData))

    return momentum_weights[0]* K.sum(abs(moments_data[0] - moments_mc[0])) + momentum_weights[1]*K.sum(abs(moments_data[1] - moments_mc[1]))

global_loss_list['moments_weighted']=moments_weighted

def combined_ccemv(y_true, y_pred):
    '''
    :param y_true: [isB, isBB, isLeptB, isC, isUDS, isG,  isData, eventweight]
    :param y_pred: [prob_isB, prob_isBB, prob_isLeptB, prob_isC, prob_isUDS, prob_isG]
    :return: combined loss function of cross entropy and moments (means and variances)
    '''

    weight_ce = 1.
    weight_m = 0.5
    weight_v = 0.1

    y_true_dlw =  y_true[:,-2:]  #domain label and eventweight

    return moments_weighted(y_true_dlw,y_pred,[weight_m,weight_v]) + weight_ce * cce_weighted_dex(y_true, y_pred)

global_loss_list['combined_ccemv']= combined_ccemv

def huberishLoss_noUnc(y_true, x_pred):
    
    
    dxrel=(x_pred - y_true)/1#(K.clip(K.abs(y_true+0.1),K.epsilon(),None))
    dxrel=K.clip(dxrel,-1e6,1e6)
    
    #defines the inverse of starting point of the linear behaviour
    scaler=2
    
    dxabs=K.abs(scaler* dxrel)
    dxsq=K.square(scaler * dxrel)
    dxp4=K.square(dxsq)
    
    lossval=dxsq / (1+dxp4) + (2*dxabs -1)/(1 + 1/dxp4)
    #K.clip(lossval,-1e6,1e6)
    
    return K.mean( lossval , axis=-1)
    


global_loss_list['huberishLoss_noUnc']=huberishLoss_noUnc



def loss_NLL(y_true, x):
    """
    This loss is the negative log likelyhood for gaussian pdf.
    See e.g. http://bayesiandeeplearning.org/papers/BDL_29.pdf for details
    Generally it might be better to even use Mixture density networks (i.e. more complex pdf than single gauss, see:
    https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.log(K.square(x_sig))  + K.square(x_pred - y_true)/K.square(x_sig)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_NLL']=loss_NLL

def loss_meansquared(y_true, x):
    """
    This loss is a standard mean squared error loss with a dummy for the uncertainty, 
    which will just get minimised to 0.
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.square(x_sig)  + K.square(x_pred - y_true)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_meansquared']=loss_meansquared


def loss_logcosh(y_true, x):
    """
    This loss implements a logcosh loss with a dummy for the uncertainty.
    It approximates a mean-squared loss for small differences and a linear one for
    large differences, therefore it is conceptually similar to the Huber loss.
    This loss here is scaled, such that it start becoming linear around 4-5 sigma
    """
    scalefactor_a=30
    scalefactor_b=0.4
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    def cosh(y):
        return (K.exp(y) + K.exp(-y)) / 2
    
    return K.mean(0.5*K.square(x_sig))   + K.mean(scalefactor_a* K.log(cosh( scalefactor_b*(x_pred - y_true))), axis=-1)
    


global_loss_list['loss_logcosh']=loss_logcosh


def loss_logcosh_noUnc(y_true, x_pred):
    """
    This loss implements a logcosh loss without a dummy for the uncertainty.
    It approximates a mean-squared loss for small differences and a linear one for
    large differences, therefore it is conceptually similar to the Huber loss.
    This loss here is scaled, such that it start becoming linear around 4-5 sigma
    """
    scalefactor_a=1.
    scalefactor_b=3.
    
    from tensorflow import where, greater, abs, zeros_like, exp
    
    dxrel=(x_pred - y_true)/(y_true+0.0001)
    def cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    
    return scalefactor_a*K.mean( K.log(cosh(scalefactor_b*dxrel)), axis=-1)
    


global_loss_list['loss_logcosh_noUnc']=loss_logcosh_noUnc

# The below is to use multiple gaussians for regression

## https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb
## these next three functions are from Axel Brando and open source, but credits need be to https://creativecommons.org/licenses/by-sa/4.0/ in case we use it

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True))+x_max
                       

global_loss_list['log_sum_exp']=log_sum_exp

def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
    - float(c) * K.log(sigma) \
    - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


global_loss_list['mean_log_Gaussian_like']=mean_log_Gaussian_like


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-2,1.))
    
    exponent = K.log(alpha) - float(c) * K.log(2 * sigma) \
    - K.sum(K.abs(K.expand_dims(y_true,2) - mu), axis=1)/(sigma)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

global_loss_list['mean_log_LaPlace_like']=mean_log_LaPlace_like

