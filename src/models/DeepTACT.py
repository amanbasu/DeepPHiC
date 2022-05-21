import tensorflow as tf

'''
TensorFlow implementation of DeepTACT (https://github.com/liwenran/DeepTACT)
'''

'''
Attentinon layer modified from the original implementation 
'''
class AttLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.init = tf.keras.initializers.HeNormal()
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = tf.Variable(
            initial_value=self.init((input_shape[-1],1)), 
            trainable=True
            )
        super(AttLayer, self).build(input_shape)

    def call(self, x):
        M = tf.math.tanh(x)
        alpha = tf.matmul(M,self.W)

        ai = tf.math.exp(alpha)
        weights = ai / tf.expand_dims(tf.reduce_sum(ai, axis=1), axis=-1)
        weighted_input = x * weights
        
        return tf.math.tanh(tf.reduce_sum(weighted_input, axis=1))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

class DeepTACT:
    def __init__(
        self, 
        filter=256, 
        dropout=0.5, 
        cell=100, 
        dense=925, 
        shape1=(2000,),                                                         # sequence features 
        shape2=(21, 1),                                                         # read features
        learning_rate=1e-4
        ):

        self.filter = filter
        self.dropout = dropout
        self.cell = cell
        self.dense = dense
        self.shape1 = shape1
        self.shape2 = shape2
        self.learning_rate = learning_rate

        # reshape features like in the original model
        self.shape1_ = (4, self.shape1[0], 1)
        self.shape2_ = (self.shape2[1], self.shape2[0], 1)

        self.model = None                                                       # TensorFlow model

        tf.random.set_seed(0)
        self.__build__()
    
    def conv_block(
        self, 
        x, 
        k=64, 
        w=40, 
        s=20
        ):

        conv = tf.keras.layers.Conv2D(k, (1, w), s, activation='relu')(x)
        pool = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(conv)
        reshape = tf.keras.layers.Reshape((-1, self.filter))(pool)

        return reshape

    def __build__(
        self
        ):

        def one_hot(x):
            return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape1)
        input2 = tf.keras.layers.Input(shape=self.shape1)
        input3 = tf.keras.layers.Input(shape=self.shape2)
        input4 = tf.keras.layers.Input(shape=self.shape2)

        # convert sequences to one-hot
        input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        reshape1 = tf.keras.layers.Reshape(self.shape1_)(
            tf.keras.layers.Permute((2, 1))(input1_oh)
            )
        reshape2 = tf.keras.layers.Reshape(self.shape1_)(
            tf.keras.layers.Permute((2, 1))(input2_oh)
            )
        reshape3 = tf.keras.layers.Reshape(self.shape2_)(
            tf.keras.layers.Permute((2, 1))(input3)
            )
        reshape4 = tf.keras.layers.Reshape(self.shape2_)(
            tf.keras.layers.Permute((2, 1))(input4)
            )

        merge1 = tf.keras.layers.Concatenate(axis=1)(
            [self.conv_block(reshape1, self.filter, 40, 20), 
            self.conv_block(reshape2, self.filter, 40, 20)]
            )
        merge2 = tf.keras.layers.Concatenate(axis=1)(
            [self.conv_block(reshape3, self.filter, 3, 1), 
            self.conv_block(reshape4, self.filter, 3, 1)]
            )
        merge3 = tf.keras.layers.Concatenate(axis=-2)([merge1, merge2])

        perm1 = tf.keras.layers.Permute((2, 1))(merge3)
        norm1 = tf.keras.layers.BatchNormalization()(perm1)
        drop1 = tf.keras.layers.Dropout(self.dropout)(norm1)

        blstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.cell, 
                return_sequences=True
                )
            )(drop1)
        attention = AttLayer()(blstm)
        norm2 = tf.keras.layers.BatchNormalization()(attention)
        drop2 = tf.keras.layers.Dropout(self.dropout)(norm2)

        dense1 = tf.keras.layers.Dense(
            self.dense, 
            kernel_initializer=tf.keras.initializers.HeNormal
            )(drop2)
        norm3 = tf.keras.layers.BatchNormalization()(dense1)
        act1 = tf.keras.layers.ReLU()(norm3)
        drop3 = tf.keras.layers.Dropout(self.dropout)(act1)

        out = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=tf.keras.initializers.HeNormal
            )(drop3)

        self.model = tf.keras.models.Model(
            inputs=[input1, input2, input3, input4],
            outputs=out
            ) 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy', 
            metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc')]
            ) 

    '''
    DeepTACT only uses sequence and read features. 
    Hence, inp5 and inp6 are ignored.
    '''
    def fit(
        self, 
        inp1,                                                                   # 1st sequence
        inp2,                                                                   # 2nd sequence
        inp3,                                                                   # 1st read
        inp4,                                                                   # 2nd read
        inp5,                                                                   # 1st distance
        inp6,                                                                   # 2nd distance
        y, 
        epochs=200, 
        batch_size=128, 
        validation_data=None
        ):

        # learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, 
            mode='auto', 
            restore_best_weights=True
            )
        validation_data = (validation_data[0][:4], validation_data[1])

        self.model.fit(
            [inp1, inp2, inp3, inp4],
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate], 
            validation_data=validation_data, 
            verbose=2
            )

    '''
    DeepTACT only uses sequence and read features. 
    Hence, inp5 and inp6 are ignored.
    '''
    def predict(
        self, 
        inp1, 
        inp2, 
        inp3, 
        inp4, 
        inp5, 
        inp6):

        return self.model.predict([inp1, inp2, inp3, inp4])

    def save_model(
        self, 
        model_name
        ):

        self.model.save(model_name)