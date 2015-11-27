import cPickle as pickle
import json
import numpy as np
import os

class DataProvider:
  def __init__(self, params):
    # Write the initilization code to load the preprocessed data and labels
    self.dataDesc = json.load(open(os.path.join('data', params['dataset'], params['dataDesc']), 'r'))
    self.in_dim = params['in_dim']
    self.data = {}
    for splt in ['train','eval','devel']:
      self.data[splt] = {}
      self.data[splt]['feat'],self.data[splt]['lab'] = self.load_data(self.dataDesc[splt+'_x'], self.dataDesc[splt+'_y'])

    self.feat_size = self.data['train']['feat'].shape[-1]
    self.phone_vocab = len(self.dataDesc['ph2bin'].keys())

  def getBatch(self, batch_size):
    return []*batch_size

  def getBatchWithContext(self):
    return []

  def getSplitSize(self, split='train'):
    return self.data[split]['feat'].shape[0] 

  def load_data(self, input_file_list, output_file_list, out_dim=24, shufdata = 1):
      """
      load partiotion
      """
      in_dim = self.in_dim
      for i in xrange(len(input_file_list)):  
          in_data = np.fromfile(input_file_list[i],dtype=np.float32,sep=' ',count=-1)
          out_data = np.fromfile(output_file_list[i],dtype=np.float32,sep=' ',count=-1)
          if i > 0:
              input_data = np.concatenate((input_data,in_data))
              output_data = np.concatenate((output_data, out_data))
          else:
              input_data = in_data
              output_data = out_data
      input_data.resize(len(input_data)/in_dim, in_dim)
      output_data.resize(len(output_data)/out_dim, out_dim)
      shfidx = np.random.permutation(input_data.shape[0]) if shufdata == 1 else np.arange(input_data.shape[0])
      input_data = input_data[shfidx,:]
      output_data = output_data[shfidx,:]
  
      return input_data, output_data

