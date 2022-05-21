import tensorflow as tf

'''
TensorFlow implementation of SPEID (https://github.com/ma-compbio/SPEID)
'''

class SPEID:
    def __init__(
        self, 
        kernel=128, 
        dropout=0.5, 
        cell=100, 
        dense=512, 
        shape=(2000),                                                           # sequence features 
        learning_rate=1e-3
        ):

        self.kernel = kernel
        self.dropout = dropout
        self.cell = cell
        self.dense = dense
        self.shape = shape
        self.learning_rate = learning_rate

        self.model = None                                                       # TensorFlow model

        tf.random.set_seed(0)
        self.__build__()
    
    def conv_block(
        self, 
        x, 
        k, 
        w=40, 
        s=1
        ):

        conv = tf.keras.layers.Conv1D(
            k,
            w, 
            s, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.L2(1e-5)
            )(x)
        pool = tf.keras.layers.MaxPool1D(pool_size=w//2, strides=w//2)(conv)

        return pool

    def __build__(
        self
        ):

        def one_hot(x):
            return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape)
        input2 = tf.keras.layers.Input(shape=self.shape)

        # convert sequences to one-hot
        input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        merge1 = tf.keras.layers.Concatenate(axis=1)(
            [self.conv_block(input1_oh, self.kernel), 
            self.conv_block(input2_oh, self.kernel)]
            )
        norm1 = tf.keras.layers.BatchNormalization()(merge1)
        drop1 = tf.keras.layers.Dropout(self.dropout)(norm1)

        blstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.cell, 
                return_sequences=True
                )
            )(drop1)
        norm2 = tf.keras.layers.BatchNormalization()(blstm)
        drop2 = tf.keras.layers.Dropout(self.dropout)(norm2)

        flatten = tf.keras.layers.Flatten()(drop2)

        dense1 = tf.keras.layers.Dense(
            self.dense, 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0), 
            kernel_regularizer=tf.keras.regularizers.L2(1e-6)
            )(flatten)
        norm3 = tf.keras.layers.BatchNormalization()(dense1)
        act1 = tf.keras.layers.ReLU()(norm3)
        drop3 = tf.keras.layers.Dropout(self.dropout)(act1)

        out = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(drop3)

        self.model = tf.keras.models.Model(inputs=[input1, input2], outputs=out) 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy', 
            metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc')]
            ) 

    '''
    SPEID only uses sequence features. Hence, inp3-inp6 are ignored.
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
        validation_data = (validation_data[0][:2], validation_data[1])
        
        self.model.fit(
            [inp1, inp2],
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate],
            validation_data=validation_data,
            verbose=2
            )

    '''
    SPEID only uses sequence features. Hence, inp3-inp6 are ignored.
    '''
    def predict(
        self, 
        inp1, 
        inp2, 
        inp3, 
        inp4, 
        inp5, 
        inp6):

        return self.model.predict([inp1, inp2])

    def save_model(
        self, 
        model_name
        ):

        self.model.save(model_name)