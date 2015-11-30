import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import SimpleRNN 
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from os import path

class RnnClassifier:
  def __init__(self, params):
    self.model = Sequential()
    print('----------Using RNN model with the below configuration----------') 
    print('nLayers:%d'%(len(params['hidden_layers'])))
    print('Layer sizes: [%s]'%(' '.join(map(str,params['hidden_layers']))))
    print('Dropout Prob: %.2f '%(params['drop_prob_encoder']))

  def build_model(self, params):
    hidden_layers = params['hidden_layers']
    input_dim = params['feat_size']
    output_dim = params['phone_vocab_size']
    drop_prob = params['drop_prob_encoder']
    self.nLayers = len(hidden_layers)

    # First Layer is the RNN layer
    self.model.add(SimpleRNN(hidden_layers[0], init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid', weights=None, truncate_gradient=-1, return_sequences=False, 
        input_dim=input_dim, input_length=None))

    # Then we add dense projection layer to map the RNN outputs to Vocab size 
    self.model.add(Dropout(drop_prob))
    self.model.add(Dense(output_dim, input_dim=hidden_layers[0], init='uniform'))
    self.model.add(Activation('softmax'))
  
    if params['solver'] == 'sgd':
      self.solver = SGD(lr=params['lr'], decay=1-params['decay_rate'], momentum=0.9, nesterov=True)
    else:  
      raise ValueError('ERROR in RNN: %s --> This solver type is not yet supported '%(params['solver']))
      
    self.model.compile(loss='categorical_crossentropy', optimizer=self.solver)
    #score = model.evaluate(test_x)
    self.f_train = self.model.train_on_batch

    return self.f_train

  def train_model(self, train_x, train_y, val_x, val_y,params):
    epoch= params['max_epochs']
    batch_size=params['batch_size']
    out_dir=params['out_dir']
    fname = path.join(out_dir, 'RNN_weights_'+params['out_file_append'] +'_{val_loss:.2f}.hdf5')
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    self.model.fit(train_x, train_y,validation_data=(val_x, val_y), nb_epoch=epoch, batch_size=batch_size, callbacks=[checkpointer])
    return fname, checkpointer.best
      

