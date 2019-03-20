import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend
from src.fabio.RandomFeatures import RFF
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, Normalizer
import delfi.distribution as dd


# Set up CPU or GPU
config = tf.ConfigProto( device_count = {'CPU': 4, 'GPU': 0})
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
backend.set_session(sess)

# Creates a MDN with elliptical components (sigmas differ in each dimension)
# Imports of the Keras library parts we will need
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, concatenate
from keras.callbacks import History
from keras.regularizers import l2
from keras.initializers import lecun_normal
from keras.objectives import mean_absolute_error
from delfi.kernel import Gauss

# Other imports
from datetime import datetime
import matplotlib.pyplot as plt

# Definition of the ELU+1 function
# With some margin to avoid problems of instability
from keras.layers.advanced_activations import ELU
from keras.callbacks import Callback, ModelCheckpoint


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))



class MDRFF(object):
    def __init__(self, ncomp=10, nfeat=100, inputd=None, cosOnly=False,
                 kernel="RBF", sigma=1, quasiRandom=True, outputd=None, weights=None):

        self.ncomp = ncomp # number of mixture components
        self.nfeat = nfeat # number of features
        self.inputd = inputd # dimensionality of the input
        self.outputd = outputd # dimensionality of the output
        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        self.sigma = sigma * np.ones(self.inputd)
        self.kernel = kernel
        self.rff = RFF(self.nfeat, self.inputd, self.sigma,
                       self.cosOnly, self.quasiRandom, self.kernel)
        self.weights = weights
        self.scaler  = StandardScaler()
        #self.scaler = MinMaxScaler()
        #self.scaler = MaxAbsScaler()
        #self.scaler = Normalizer(norm='l2')
        #self.scaler = QuantileTransformer(output_distribution='normal', random_state=0)


        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e
        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True)) + x_max


        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, 2 * self.outputd + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            sigma = components[:, self.outputd:2 * self.outputd, :]
            alpha = components[:, 2 * self.outputd, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)
            sigma = K.clip(sigma, 1e-8, 1e8)

            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalDiag(
                loc=K.reshape(mu, [-1, self.ncomp, self.outputd]),
                scale_diag=K.reshape(sigma, [-1, self.ncomp, self.outputd])))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res


        # This returns a tensor
        inputs = Input(shape=(self.nfeat,))

        # a layer instance is callable on a tensor, and returns a tensor
        #nn = Dense(self.nunits[0], activation='tanh', kernel_initializer='RandomNormal')(inputs)
        #nn = Dropout(0.05)(nn)

        #for i in range(self.nhidden - 2):
        #    nn = Dense(self.nunits[i+1], activation='tanh', kernel_initializer='RandomNormal')(nn)
        #    nn = Dropout(0.05)(nn)

        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer='RandomUniform',
                       name='FC_mus')(inputs)
        FC_sigmas = Dense(units=self.outputd * self.ncomp,
                          activation=elu_modif,
                          kernel_initializer='Ones',
                          kernel_regularizer=l2(1e-3),
                          name='FC_sigmas')(inputs)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer='Ones',
                          name='FC_alphas')(inputs)

        output = concatenate([FC_mus, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False):
        lossHistory = LossHistory()

        # checkpoint
        if save:
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            # Fix nan values in the dataset
            ind = np.isnan(x_data)
            x_data[ind] = 0.
            x_dataS = self.scaler.fit_transform(x_data.T)
            x_feat = self.rff.toFeatures(x_dataS)
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)

            self.model.fit(x_feat, y_data.T,
                           sample_weight=w,
                           epochs=nepoch,
                           validation_split=0.1,
                           callbacks=[lossHistory, checkpoint])
        else:
            start_time = datetime.now()
            # Fix nan values in the dataset
            ind = np.isnan(x_data)
            x_data[ind] = 0.
            x_dataS = self.scaler.fit_transform(x_data.T)
            x_feat = self.rff.toFeatures(x_dataS)
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)

            self.model.fit(x_feat, y_data.T,
                           sample_weight=w,
                           epochs=nepoch,
                           validation_split=0.1,
                           callbacks=[lossHistory],
                           verbose=0)


        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        x_testS = self.scaler.transform(x_test.T)
        x_feat = self.rff.toFeatures(x_testS)
        y_pred = self.model.predict(x_feat)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        return y_pred
    # Prediction function returning a delfi MoG
    def predict_mog_from_stats(self, ntest=1, alpha_pred=None, mu_pred=None,sigma_pred=None):
        alpha_pred = np.array(alpha_pred).reshape(-1, self.ncomp)
        mu_pred = np.array(mu_pred).reshape(-1, self.ncomp, self.outputd)
        sigma_pred = np.array(sigma_pred).reshape(-11, self.ncomp, self.outputd)
        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            di = np.diag_indices(self.outputd) #diagonal indices
            for i in range(self.ncomp):
                tmp = np.zeros((self.outputd, self.outputd))
                tmp[di] = sigma_pred[pt, i, :]**2
                Ss.append(tmp)
            mog.append(dd.MoG(a=a, ms=ms, Ss=Ss))
        return mog[0]
    # Prediction function returning a delfi MoG

    def predict_mog(self, x_test):
        start_time = datetime.now()
        x_testS = self.scaler.transform(x_test)
        x_feat = self.rff.toFeatures(x_testS)
        y_pred = self.model.predict(x_feat)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        ntest, dim = x_test.shape #test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, 2 * self.outputd + 1, self.ncomp])
        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:2 * self.outputd, :]
        alpha_pred = comp[:, 2 * self.outputd, :]
        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp, self.outputd])
        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            di = np.diag_indices(self.outputd) #diagonal indices
            for i in range(self.ncomp):
                tmp = np.zeros((self.outputd, self.outputd))
                tmp[di] = sigma_pred[pt, i, :]**2
                Ss.append(tmp)
            mog.append(dd.MoG(a=a, ms=ms, Ss=Ss))
        return mog


class MDNN(object):
    def __init__(self, ncomp=10, nhidden=2, nunits=[24, 24], inputd=None, outputd=None):

        self.ncomp = ncomp  # number of mixture components
        self.nhidden = nhidden  # number of hidden layers
        self.nunits = nunits  # number of units per hidden layer (integer or array)
        self.inputd = inputd  # dimensionality of the input
        self.outputd = outputd  # dimensionality of the output

        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e

        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max

        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, 2 * self.outputd + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            sigma = components[:, self.outputd:2 * self.outputd, :]
            alpha = components[:, 2 * self.outputd, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)

            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=K.reshape(mu, [-1, self.ncomp, self.outputd]),
                    scale_diag=K.reshape(sigma, [-1, self.ncomp, self.outputd])))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res

        # This returns a tensor
        inputs = Input(shape=(self.inputd,))

        # Initializer with a particular seed
        initializer = lecun_normal(seed=1.)

        # a layer instance is callable on a tensor, and returns a tensor
        nn = Dense(self.nunits[0], activation='tanh', kernel_initializer=initializer)(inputs)
        #nn = Dropout(0.05)(nn)

        for i in range(self.nhidden - 2):
            nn = Dense(self.nunits[i + 1], activation='tanh', kernel_initializer=initializer)(nn)
            #nn = Dropout(0.05)(nn)

        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer=initializer,
                       name='FC_mus')(nn)
        FC_sigmas = Dense(units=self.outputd * self.ncomp,
                          activation=elu_modif,
                          kernel_initializer=initializer,
                          name='FC_sigmas')(nn)  # K.exp, W_regularizer=l2(1e-3)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer=initializer,
                          name='FC_alphas')(nn)

        output = concatenate([FC_mus, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False):
        lossHistory = LossHistory()

        # checkpoint
        if save:
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            self.model.fit(x_data.T, y_data.T, epochs=nepoch, #validation_split=0.1,
                           callbacks=[lossHistory, checkpoint], verbose=0)
        else:
            start_time = datetime.now()
            self.model.fit(x_data.T, y_data.T, epochs=nepoch, callbacks=[lossHistory], verbose=0)

        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        y_pred = self.model.predict(x_test.T)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        return y_pred

    # Prediction function returning a delfi MoG
    def predict_mog(self, x_test):
        start_time = datetime.now()
        y_pred = self.model.predict(x_test)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        ntest, dim = x_test.shape  # test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, 2 * self.outputd + 1, self.ncomp])
        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:2 * self.outputd, :]
        alpha_pred = comp[:, 2 * self.outputd, :]
        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp, self.outputd])
        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            di = np.diag_indices(self.outputd)  # diagonal indices
            for i in range(self.ncomp):
                tmp = np.zeros((self.outputd, self.outputd))
                tmp[di] = sigma_pred[pt, i, :] ** 2
                Ss.append(tmp)
            mog.append(dd.MoG(a=a, ms=ms, Ss=Ss))
        return mog
    # Prediction function returning a delfi MoG

    # Prediction function returning a delfi MoG


