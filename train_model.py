import numpy as np
import theano
import argparse
import json
from utils.solver import Solver 
from utils.dataprovider import DataProvider
#from mlpmodel import mlpClassifier
#from rnnmodel import rnnClassifier

def getModelObj(params):
  if params['model_type'] == 'MLP':
    mdl = mlpClassifier(params) 
  elif params['model_type'] == 'RNN':  
    mdl = rnnClassifier(params) 
  else:
    raise ValueError('ERROR: %s --> This model type is not yet supported'%(params['model_type']))

def main(params):
  
  # main training and validation loop goes here
  # This code should be independent of which model we use
  batch_size = params['batch_size']
  max_epochs = params['max_epochs']
  
  # fetch the data provider object
  dp = DataProvider(params)
  
  # Get the solver object
  solver = Solver(params['solver'])
  
  
  ## Add the model intiailization code here
  modelObj = getModelObj(params)
  
  
  
  # Now let's build a gradient computation graph and rmsprop update mechanism
  #grads = tensor.grad(cost, wrt=model.values())
  #lr = tensor.scalar(name='lr',dtype=config.floatX)
  #f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, grads,
  #                                   inp_list, cost, params)
  
  num_frames_total = dp.getSplitSize('train')
  num_iters_one_epoch = num_frames_total/ batch_size
  max_iters = max_epochs * num_iters_one_epoch
  
  for it in xrange(max_iters):
    batch = dp.getBatch(batch_size)
    
    #cost = f_grad_shared(inp_list)
    #f_update(params['learning_rate'])

    #Save model periodically



if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  # IO specs
  parser.add_argument('-d','--dataset', dest='dataset', type=str, default='mvNorm', help='Which file should we use for read the MFCC features')
  parser.add_argument('--dataset_desc', dest='dataDesc', type=str, default='dataset.json', help='Which file should we use for read the MFCC features')
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='data/default_feats.p', help='Which file should we use for read the MFCC features')
  parser.add_argument('--output_file_append', dest='out_file_append', type=str, default='dummyModel', help='String to append to the filename of the trained model')
  
  # Learning related parmeters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  
  # Solver related parameters
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver types supported: rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  
  
  # Model architecture related parameters
  parser.add_argument('--model_type', dest='model_type', type=str, default='MLP', help='Can take values MLP, RNN or LSTM')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1, help='enable or disable dropout')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  #main(params)
