#@title Network Manager
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from model_set import Custom1DCNN

# model.fit(X_train, Y_train,
#           epochs=10,
#           validation_data=(X_test, Y_test),
#           callbacks=[PlotLossesKeras()],
#           verbose=0)

class NetworkManager:
    def __init__(self, dataset, epochs, batchsize, learning_rate):
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = learning_rate

    def get_reward(self, save_path = None):
        '''
        '''
        if tf.config.experimental.list_physical_devices('GPU'):
            device = tf.test.gpu_device_name()
        else:
            device = 'CPU:0'
        
        if save_path == None:
            save_path = 'default/100data_point'
        
        with tf.device(device):
            X_train, y_train, x_val, y_val = self.dataset
            checkpoint = ModelCheckpoint( # set model saving checkpoints
                save_path, # set path to save model weights
                monitor='val_loss', # set monitor metrics
                verbose=1, # set training verbosity
                save_best_only=True, # set if want to save only best weights
                save_weights_only=False, # set if you want to save only model weights
                mode='auto', # set if save min or max in metrics
                period=1 # interval between checkpoints
                )
            opt = tf.keras.optimizers.Adam(learning_rate = self.lr)
            loss = tf.keras.losses.CategoricalCrossentropy()
            model = Custom1DCNN(inp_shape=(X_train.shape[1], X_train.shape[2]), nbr_classes=y_train.shape[1])

            model.compile(optimizer = opt, loss=loss , metrics=['accuracy'])

            hist = model.fit(X_train, y_train, 
                             batch_size= self.batchsize, epochs = self.epochs, verbose = True, 
                             validation_data = (x_val,y_val),callbacks=[checkpoint])

            testLoss, testAcc = model.evaluate(x_val, y_val)
            return hist, model, testAcc 