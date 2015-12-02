import numpy as np
import theano
import argparse
import json
import os
from utils.dataprovider import DataProvider
from utils.utils import getModelObj
import re
import codecs
import struct

def dump_lna(probs, lna_file, n_phones=24):
    with open(lna_file, 'wb') as f:
        f.write(struct.pack(">I", n_phones))
        f.write(struct.pack("B", 2))
        intProbs = map(int,-1820.0 * np.maximum(np.log(probs),-7.0) + 0.5)
        f.write(struct.pack('>' +len(intProbs)*'H', *intProbs))

def main(params):
  # check if having a model_list  
  if params['model_list'] != None:
    with open(params['model_list']) as f:
      model_file_list = f.readlines()
  else:
    model_file_list = [(params['saved_model'])]
  
  # check dp loaded or not to load it once
  dp_loaded = False
  for m in model_file_list:
    m = re.sub("\n","", m)
    cv = json.load(open(m,'r'))
    cv_params = cv['params']
    
    if params['dataset'] != None:
        cv_params['dataset'] = params['dataset']
        cv_params['dataset_desc'] = params['dataset_desc']
    if not dp_loaded:
      dp_loaded = True
      dp = DataProvider(cv_params)
    cv_params['feat_size'] = dp.feat_size
    cv_params['phone_vocab_size'] = dp.phone_vocab
    
    
    # Get the model object and build the model Architecture
    modelObj = getModelObj(cv_params)
    f_train = modelObj.build_model(cv_params)
    modelObj.model.load_weights(cv['weights_file'])
    
    inpt_x, inpt_y = dp.get_data_array(cv_params['model_type'],[params['split']],cntxt = cv_params['context'])

    #predOut = modelObj.model.predict_classes(inpt_x, batch_size=100)
    #accuracy =  100.0*np.sum(predOut == inpt_y.nonzero()[1]) / predOut.shape[0]
    #print('Accuracy of %s the %s set is %0.2f'%(params['saved_model'], params['split'],accuracy))

    if params['dump_lna_dir'] != None:
        spt = params['split']
        ph2bin = dp.dataDesc['ph2bin']
        phoneList = ['']*len(ph2bin)
        for ph in ph2bin:
            phoneList[ph2bin[ph].split().index('1')] = ph
        phones_targ = [l.strip() for l in codecs.open(params['lna_ph_order'], encoding='utf-8')]
        assert(set(phones_targ) == set(phoneList))
        shuffle_order = np.zeros(len(phones_targ),dtype=np.int32)
        for i,ph in enumerate(phoneList):
            shuffle_order[i] = phones_targ.index(ph)
        ## Now for evert utterance sample predict probabilities and dump lna files
        for i,inp_file in enumerate(dp.dataDesc[spt+'_x']): 
            lna_file = os.path.join(params['dump_lna_dir'], os.path.basename(inp_file).split('.')[0]+'.lna')
            inpt_x,_ = dp.get_data_array(cv_params['model_type'],[params['split']],cntxt = cv_params['context'], shufdata=0, idx = i)
            probs = modelObj.model.predict(inpt_x, batch_size=100)
            dump_lna(probs[:,shuffle_order].flatten(), lna_file, probs.shape[1])
            print lna_file


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  # IO specs
  parser.add_argument('-m','--saved_model', dest='saved_model', type=str, default='', help='input the saved model json file to evluate on')
  parser.add_argument('-s','--split', dest='split', type=str, default='eval', help='which data split to evaluate on')
  
  # Provide these only if evaluating on a dataset other than what the model was trained on
  parser.add_argument('-d','--dataset', dest='dataset', type=str, default=None, help='Which file should we use for read the MFCC features')
  parser.add_argument('--dataset_desc', dest='dataDesc', type=str, default='dataset.json', help='Which file should we use for read the MFCC features')
  
  parser.add_argument('--dump_lna_dir', dest='dump_lna_dir', type=str, default=None, help='Should we dump lna files ?')
  parser.add_argument('--lna_ph_order', dest='lna_ph_order', type=str, default='list_monophones', help='Phone order to follow')
  
  # models list file
  parser.add_argument('--model_list', dest='model_list', type=str, default=None,\
  help='Text file containing of model files list to evaluate')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
