import tensorflow as tf

'''
TensorFlow implementation of DeepMILO (https://github.com/khuranalab/DeepMILO)
'''

class DeepMILO:
    def __init__(
        self, 
        filter=[64, 128], 
        dilation=[1, 3, 7], 
        dropout=0.15, 
        leaky=0.2,
        dense=[256, 128], 
        cell=[128, 128], 
        shape=(2000,),                                                          # sequence features
        learning_rate=1e-3
        ):

        self.filter = filter
        self.dilation = dilation
        self.dropout = dropout
        self.leaky = leaky
        self.dense = dense
        self.cell = cell
        self.shape = shape
        self.learning_rate = learning_rate

        self.init = tf.keras.initializers.HeNormal
        self.model = None                                                       # TensorFlow model
        
        tf.random.set_seed(0)
        self.__build__()
 
    def __conv_block__(
        self,
        x, 
        k=512, 
        w=(5, 1), 
        s=1, 
        dilation=1
        ):

        conv = tf.keras.layers.Conv2D(
            k, 
            w, 
            s, 
            padding='same', 
            dilation_rate=dilation
            )(x)
        norm = tf.keras.layers.BatchNormalization()(conv)
        relu = tf.keras.layers.LeakyReLU(self.leaky)(norm)
        pool = tf.keras.layers.MaxPooling2D((20, 1), strides=(20, 1))(relu)
        drop = tf.keras.layers.Dropout(self.dropout)(pool)
        flat = tf.keras.layers.Flatten()(drop)

        return flat

    def __conv_net__(
        self, 
        x, 
        w, 
        s
        ):

        conv1 = tf.keras.layers.Conv2D(self.filter[0], w, s)(x)
        norm = tf.keras.layers.BatchNormalization()(conv1)
        relu = tf.keras.layers.LeakyReLU(self.leaky)(norm)
        drop = tf.keras.layers.Dropout(self.dropout)(relu)

        conv2 = self.__conv_block__(
            drop, 
            k=self.filter[1], 
            dilation=self.dilation[0]
            )
        conv3 = self.__conv_block__(
            drop, 
            k=self.filter[1], 
            dilation=self.dilation[1]
            )
        conv4 = self.__conv_block__(
            drop, 
            k=self.filter[1], 
            dilation=self.dilation[2]
            )

        merge = tf.keras.layers.Concatenate(axis=-1)([conv2, conv3, conv4])

        return merge

    def __blstm_net__(
        self, 
        x
        ):

        blstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.cell[0], 
                return_sequences=True, 
                dropout=self.dropout
                )
            )(x)
        blstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.cell[1], 
                return_sequences=True, 
                dropout=self.dropout
                )
            )(blstm1)
        tdist = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                1, 
                activation='sigmoid', 
                kernel_initializer=self.init
                )
            )(blstm2)
        flat = tf.keras.layers.Flatten()(tdist)

        return flat

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

        merge = tf.keras.layers.Concatenate(axis=-1)([input1_oh, input2_oh])

        feat1 = self.__conv_net__(
            tf.expand_dims(merge, axis=-1), 
            w=(17, 4), 
            s=(1, 4)
            )
        feat2 = self.__blstm_net__(merge)

        dense1 = tf.keras.layers.Dense(
            self.dense[0], 
            kernel_initializer=self.init
            )(feat1)
        norm1 = tf.keras.layers.BatchNormalization()(dense1)
        relu1 = tf.keras.layers.LeakyReLU(self.leaky)(norm1)
        drop1 = tf.keras.layers.Dropout(self.dropout)(relu1)

        dense2 = tf.keras.layers.Dense(
            self.dense[1], 
            kernel_initializer=self.init
            )(drop1)
        norm2 = tf.keras.layers.BatchNormalization()(dense2)
        relu2 = tf.keras.layers.LeakyReLU(self.leaky)(norm2)
        drop2 = tf.keras.layers.Dropout(self.dropout)(relu2)

        dense3 = tf.keras.layers.Dense(
            self.dense[0], 
            kernel_initializer=self.init
            )(feat2)
        norm3 = tf.keras.layers.BatchNormalization()(dense3)
        relu3 = tf.keras.layers.LeakyReLU(self.leaky)(norm3)
        drop3 = tf.keras.layers.Dropout(self.dropout)(relu3)

        dense4 = tf.keras.layers.Dense(
            self.dense[1], 
            kernel_initializer=self.init
            )(drop3)
        norm4 = tf.keras.layers.BatchNormalization()(dense4)
        relu4 = tf.keras.layers.LeakyReLU(self.leaky)(norm4)
        drop4 = tf.keras.layers.Dropout(self.dropout)(relu4)
        
        out1 = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=self.init
            )(drop2)
        out2 = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=self.init
            )(drop4)

        out = tf.reduce_mean([out1, out2], axis=0)

        self.model = tf.keras.models.Model(inputs=[input1, input2], outputs=out) 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]) 

    '''
    DeepMILO only uses sequence features. Hence, inp3-inp6 are ignored.
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
    DeepMILO only uses sequence features. Hence, inp3-inp6 are ignored.
    '''
    def predict(
        self, 
        inp1, 
        inp2, 
        inp3, 
        inp4, 
        inp5, 
        inp6
        ):

        return self.model.predict([inp1, inp2])

    def save_model(
        self, 
        model_name
        ):
        
        self.model.save(model_name)