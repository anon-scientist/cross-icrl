import tensorflow as tf
from gazebo_sim.utils.logger import log
def build_dqn_models( n_actions, input_dims, fc1_dims, fc2_dims, lr, target_network="no", ckpt=None, last_activation = None):
        out_size = n_actions
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_dims))
        model.add(tf.keras.layers.Flatten());
        model.add(tf.keras.layers.Dense(fc1_dims, activation="relu"));
        model.add(tf.keras.layers.Dense(fc2_dims, activation="relu"));
        model.add(tf.keras.layers.Dense(out_size, activation=last_activation));
        optimizer=tf.optimizers.Adam(learning_rate=lr) ;
        model.compile(optimizer=optimizer)
        model.summary()
        log(target_network, "WTF!!!") ;
        if target_network=="yes":
                target_model = tf.keras.models.clone_model(model)
                target_model.summary()
        else:
                target_model = None

        return model, target_model

class DuelingDQN(tf.keras.Model):
    ''' Q(s,a) = V(s) + (A(s,a) - (1 / |A|) *\sum_{a'} A(s,a)) '''

    def __init__(self, name, n_actions, input_dims, train_batch_size, fc1_dims, fc2_dims, last_activation = None):
        super(DuelingDQN, self).__init__(name=name)
        self.n_actions = n_actions
        self.input_dims = list(input_dims)
        self.train_batch_size = train_batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputs = tf.keras.layers.InputLayer(input_shape=self.input_dims, batch_size=self.train_batch_size)
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(self.fc1_dims, activation="relu", name='fc1')
        self.dense_2 = tf.keras.layers.Dense(self.fc2_dims, activation="relu", name='fc2')
        self.V = tf.keras.layers.Dense(1, activation=None, name='V_head')
        self.A = tf.keras.layers.Dense(self.n_actions, activation=last_activation, name='A_head')

    def get_config(self):
        return {
            "name": self.name,
            "n_actions": self.n_actions,
            "input_dims": self.input_dims,
            "train_batch_size": self.train_batch_size,
            "fc1_dims": self.fc1_dims,
            "fc2_dims": self.fc2_dims
        }

    def call(self, states):
        _in = self.inputs(states)
        x = self.flat(_in)
        x = self.dense_1(x)
        x = self.dense_2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))

        return Q

def build_dueling_models(n_actions, input_dims, fc1_dims, fc2_dims, lr, train_batch_size, last_activation = None):
    optimizer=tf.optimizers.Adam(learning_rate=lr) ;
    model = DuelingDQN("online_model", n_actions, input_dims, train_batch_size, fc1_dims, fc2_dims, last_activation = last_activation)
    model.compile(run_eagerly=True, optimizer=optimizer)
    model.build(input_shape=model.input_dims)
    model.call(tf.keras.layers.Input(shape=model.input_dims)) # to init graph
    target_model = tf.keras.models.clone_model(model)
    target_model.build(input_shape=model.input_dims)
    target_model.call(tf.keras.layers.Input(shape=model.input_dims))
    target_model._name = "frozen_model"

    model.summary()
    target_model.summary()

    return model, target_model