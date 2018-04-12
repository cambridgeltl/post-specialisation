import ConfigParser
import sys, os, codecs, operator, time
reload(sys)
sys.setdefaultencoding('utf8')

import keras
import tensorflow as tf
import numpy as np
import theano
import random

from sklearn.preprocessing import normalize

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation
from keras import losses
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

"""UTILITIES: ACTIVATION FUNCTION DEFS"""
def swish(x):
        return (x*K.sigmoid(x))

get_custom_objects().update({'swish': Activation(swish)})
lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.5)
prelu = keras.layers.advanced_activations.PReLU(
    alpha_initializer='zero', weights=None)
elu = keras.layers.advanced_activations.ELU(alpha=1.0)

tf.logging.set_verbosity(tf.logging.ERROR)

class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for a
    Post-Specialisation experiment run
    """

    def __init__(self, config_filepath):
        """
        To initialise the class, we need to supply the config file,
        which contains the location of the pretrained (distributional)
        word vectors, the location of training data (after initial
        specialiastion), as well as the hyperparameters of the
        Post-Specialisation procedure (as detailed in the NAACL-HLT 2018 paper).
        """
        self.config = ConfigParser.RawConfigParser()
        try:
            self.config.read(config_filepath)
        except:
            print "Couldn't read config file from", config_filepath
            return None
        print "Loading full distributional vectors..."
        distributional_vectors_filepath = self.config.get(
            "data", "distributional_vectors"
        )

        try:
            self.output_filepath = self.config.get("data", "output_filepath")
        except:
            self.output_filepath = "results/final_vectors.txt"

        # load initial distributional word vectors.
        self.distributional_vectors = load_word_vectors(
            distributional_vectors_filepath
        )

        if not self.distributional_vectors:
            return

        # Now load the training data
        xtrain_distrib_path = self.config.get(
            "data", "distributional_training_data")
        xtrain_spec_path = self.config.get(
            "data", "specialised_training_data")

        print "Loading training data..."
        xtrain_distrib = load_word_vectors(xtrain_distrib_path)
        xtrain_spec = load_word_vectors(xtrain_spec_path)
        # Get the set of seen words (all others are unseen)
        self.seen_words = set(xtrain_spec.keys())
        self.xtrain_distrib = xtrain_distrib
        self.xtrain_spec = xtrain_spec

        # And now prepare the actual training data
        self.xtrain_distrib_items = []
        self.xtrain_spec_items = []
        for word in self.seen_words:
            self.xtrain_distrib_items.append(xtrain_distrib[word])
            self.xtrain_spec_items.append(xtrain_spec[word])
        self.xtrain_distrib_items = np.asarray(self.xtrain_distrib_items)
        self.xtrain_spec_items = np.asarray(self.xtrain_spec_items)

        # Load the experiment hyperparameters now:
        self.load_experiment_hyperparameters()

        self.embedding_size = random.choice(
            self.distributional_vectors.values()).shape[0]
        self.all_words = self.distributional_vectors.keys()
        self.vocabulary_size = len(self.all_words)
        # Now initialise the model
        self.initialise_model()


    def load_experiment_hyperparameters(self):
        """
        This method loads/sets the hyperparameters of the Post-Specialisation
        """
        # 1. Set the number of hidden layers and dimensionality
        self.number_hidden_layers = int(self.config.getfloat(
            "hyperparameters", "number_hidden_layers"))
        self.hidden_layer_size = int(self.config.getfloat(
            "hyperparameters", "hidden_layer_size"))
        # 2. Now decide on the actual activation function (default = relu)
        activation_function_str = self.config.get(
            "hyperparameters", "activation").lower()
        self.activation_function = 'relu'
        if activation_function_str == "swish":
            self.activation_function = swish
        elif activation_function_str in ["lrelu", "leaky relu", "leakyrelu"]:
            self.activation_function = lrelu
        elif activation_function_str == "prelu":
            self.activation_function = prelu
        elif activation_function_str == "elu":
            self.activation_function = elu
        else:
            self.activation_function = 'relu'
        # 3. Batch size and number of epochs/iterations
        self.max_epochs = int(self.config.getfloat(
            "hyperparameters", "max_epochs"))
        self.batch_size = int(self.config.getfloat(
            "hyperparameters", "batch_size"))
        # 4. Parameters for the max-margin loss
        self.margin = self.config.getfloat(
            "hyperparameters", "margin")
        self.negative_samples = int(self.config.getfloat(
            "hyperparameters", "negative_samples"))
        # 5. Finally, decide on the mode (fixed or all); default is fixed
        self.mode = self.config.get(
            "hyperparameters", "mode")
        if self.mode != "fixed" and self.mode != "all":
            self.mode = "fixed"

        self.initalisation = 'glorot_normal'


    def max_margin(self, y_true, y_pred):
        """
        This method computes max-margin loss
        """
        cost = 0.0
        for i in xrange(self.negative_samples):
            new_true = tf.random_shuffle(y_true)
            cost += K.maximum(
                self.margin - y_true * y_pred + new_true * y_pred, 0.)

        return K.mean(cost, axis=-1)


    def initialise_model(self):
        """
        This method initialises the model (based on the chosen activation,
        number of hidden layers, and batch size)
        """
        self.model = Sequential()
        input_layer = Input(shape=(self.embedding_size,))
        self.model.add(Dense(
            self.hidden_layer_size,
            input_dim=self.embedding_size,
            kernel_initializer=self.initalisation)
        )
        self.model.add(Activation(self.activation_function))
        for i in range(1,self.number_hidden_layers):
            self.model.add(Dense(
                self.hidden_layer_size,
                kernel_initializer=self.initalisation)
            )
            self.model.add(Activation(self.activation_function))

        # Output layer needs the activation removed
        self.model.add(Dense(
            self.embedding_size, kernel_initializer=self.initalisation)
        )


    def post_specialisation(self):
        """
        This method does the actual training
        """
        self.model.compile(optimizer='adam', loss=self.max_margin)

        es = [keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1, mode='auto')]

        self.model.fit(self.xtrain_distrib_items, self.xtrain_spec_items,
                       epochs=self.max_epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       verbose=True,
                       validation_split=0.1,
                       callbacks = es)

        # The last step is to map everything and output the vector space
        os.system("mkdir -p results")
        fenc = open(self.output_filepath, "w")
        counter = 0

        for item in self.all_words:
            counter += 1
            if counter % 5000 == 0:
                print "Mapped: ", str(counter), "/", str(self.vocabulary_size)
            seen_vector = None
            if item in self.seen_words:
                seen_vector_a = np.array(self.xtrain_spec[item],dtype='float32')
                seen_vector_n = normalize(seen_vector_a.reshape(1,-1),norm='l2',axis=1)
                seen_vector = np.ndarray.tolist(seen_vector_n)

            key = item.strip()
            vector = [self.distributional_vectors[key]]
            vector = np.asarray(vector)
            # Final transformation
            encoded_vector_a = self.model.predict(vector)
            encoded_vector_nn = encoded_vector_a[0]
            # Now normalize the vector
            encoded_vector_n = normalize(encoded_vector_nn.reshape(1,-1), norm='l2', axis=1)
            encoded_vector = np.ndarray.tolist(encoded_vector_n)

            if item in self.seen_words:
                if self.mode == "fixed":
                    encstrfixed = str(key) + " " + " ".join(map(str,seen_vector[0])) + "\n"
                    fenc.write(encstrfixed)
                else:
                    encstrall = str(key) + " "  + " ".join(map(str,encoded_vector[0])) + "\n"
                    fenc.write(encstrall)
            else:
                encstrfixed = str(key) + " "  + " ".join(map(str,encoded_vector[0])) + "\n"
                fenc.write(encstrfixed)
        fenc.close()


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the
    word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector
    dimensionality.
    """
    print "Loading vectors from", file_destination
    input_dic = {}

    with open(file_destination, "r") as in_file:
        lines = in_file.readlines()

    in_file.close()

    for line in lines:
        item = line.strip().split()
        dkey = item.pop(0)
        vector = np.array(item, dtype='float32')
        norm = np.linalg.norm(vector)
        input_dic[dkey] = vector/norm

    print len(input_dic), "vectors loaded from", file_destination

    return input_dic

def run_experiment(config_filepath):
    """
    This method runs the experiment, first parsing the config file, and then
    running the model
    """
    current_experiment = ExperimentRun(config_filepath)

    current_experiment.post_specialisation()

    os.system("mkdir -p results")

    #print_word_vectors(current_experiment.word_vectors,
    #                   current_experiment.output_filepath)

def main():
    """
    The user can provide the location of the config file as an argument.
    If no location is specified, the default config file
    (experiment_parameters.cfg) is used.
    """
    try:
        config_filepath = sys.argv[1]
    except:
        print "\nUsing the default config file: experiment_parameters.cfg\n"
        config_filepath = "experiment_parameters.cfg"

    run_experiment(config_filepath)


if __name__=='__main__':
    main()
