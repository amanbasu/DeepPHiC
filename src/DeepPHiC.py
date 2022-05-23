import tensorflow as tf

class DeepPHiC:
    def __init__(
        self, 
        filter=[64, 32, 16, 16], 
        dilation=[1, 2, 3, 4], 
        dropout=0.2, 
        dense=256, 
        shape1=(2000,),                                                         # sequence features 
        shape2=(21, 1),                                                         # read features 
        shape3=(1),                                                             # distance features 
        learning_rate=0.001
        ):

        self.filter = filter
        self.dilation = dilation
        self.dropout = dropout
        self.dense = dense
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.learning_rate = learning_rate

        self.model = None                                                       # TensorFlow model
        
        tf.random.set_seed(0)
        self.__build__()
    
    def __conv_block__(
        self, 
        x, 
        k, 
        w, 
        s, 
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
        relu = tf.keras.layers.PReLU()(norm)
        drop = tf.keras.layers.Dropout(self.dropout)(relu)

        return drop

    # inspired by DenseNet
    def __conv_net__(
        self, 
        x, 
        w=3, 
        s=[1, 1, 1, 1]
        ):

        conv1 = self.__conv_block__(
            x, 
            self.filter[0], 
            w=w, 
            s=s[0], 
            dilation=self.dilation[0]
            )

        conv2 = self.__conv_block__(
            conv1, 
            self.filter[1], 
            w=w, 
            s=s[1], 
            dilation=self.dilation[1]
            )
        merge2 = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])

        conv3 = self.__conv_block__(
            merge2, 
            self.filter[2], 
            w=w, 
            s=s[2], 
            dilation=self.dilation[2]
            )
        merge3 = tf.keras.layers.Concatenate(axis=-1)([merge2, conv3])

        conv4 = self.__conv_block__(
            merge3, 
            self.filter[3], 
            w=w, 
            s=s[3], 
            dilation=self.dilation[3]
            )
        merge4 = tf.keras.layers.Concatenate(axis=-1)([merge3, conv4])

        return merge4

    def __build__(
        self
        ):

        def one_hot(x):
            return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape1)
        input2 = tf.keras.layers.Input(shape=self.shape1)
        input3 = tf.keras.layers.Input(shape=self.shape2)
        input4 = tf.keras.layers.Input(shape=self.shape2)
        input5 = tf.keras.layers.Input(shape=self.shape3)
        input6 = tf.keras.layers.Input(shape=self.shape3)

        # convert sequences to one-hot
        input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        merge1 = tf.keras.layers.Concatenate(axis=-1)([input1_oh, input2_oh])
        merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        feat1 = self.__conv_net__(
            tf.expand_dims(merge1, axis=-1), 
            w=(20, 4), 
            s=[(10, 4), 1, 1, 1]
            )
        feat2 = self.__conv_net__(
            tf.expand_dims(merge2, axis=-1), 
            w=(3, 1), 
            s=[(1, 1), 1, 1, 1]
            )

        merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        conv = self.__conv_block__(merge4, k=self.dense, w=(4, 4), s=(4, 4))
        reshape = tf.keras.layers.Permute((2, 3, 1))(conv)
        pool = tf.keras.layers.GlobalAveragePooling2D()(reshape)

        merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3])

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(merge5)
        out = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(dense)

        self.model = tf.keras.models.Model(
            inputs=[input1, input2, input3, input4, input5, input6], 
            outputs=out
            ) 
        optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optim, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc')]
            ) 

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

        self.model.fit(
            [inp1, inp2, inp3, inp4, inp5, inp6], 
            y, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate], 
            validation_data=validation_data, 
            verbose=2
            )

    def predict(
        self, 
        inp1, 
        inp2, 
        inp3, 
        inp4, 
        inp5, 
        inp6):

        return self.model.predict([inp1, inp2, inp3, inp4, inp5, inp6])

    def save_model(
        self, 
        model_name
        ):

        self.model.save(model_name)

'''
Fusion of global and private features for Multi-task learning
'''
class DeepPHiCFusion:
    def __init__(
        self, 
        filter=[64, 32, 16, 16], 
        dilation=[1, 2, 3, 4], 
        dropout=0.2, 
        dense=256, 
        shape1=(2000,),                                                         # sequence features 
        shape2=(21, 1),                                                         # read features 
        shape3=(1),                                                             # distance features 
        shape4=(58),                                                            # global features 
        learning_rate=0.001
        ):

        self.filter = filter
        self.dilation = dilation
        self.dropout = dropout
        self.dense = dense
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.shape4 = shape4
        self.learning_rate = learning_rate

        self.model = None                                                       # TensorFlow model
        
        tf.random.set_seed(0)
        self.__build__()
    
    def __conv_block__(
        self, 
        x, 
        k, 
        w, 
        s, 
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
        relu = tf.keras.layers.PReLU()(norm)
        drop = tf.keras.layers.Dropout(self.dropout)(relu)

        return drop

    # inspired by DenseNet
    def __conv_net__(
        self, 
        x, 
        w=3, 
        s=[1, 1, 1, 1]
        ):

        conv1 = self.__conv_block__(
            x, 
            self.filter[0], 
            w=w, 
            s=s[0], 
            dilation=self.dilation[0]
            )

        conv2 = self.__conv_block__(
            conv1, 
            self.filter[1], 
            w=w, 
            s=s[1], 
            dilation=self.dilation[1]
            )
        merge2 = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])

        conv3 = self.__conv_block__(
            merge2, 
            self.filter[2], 
            w=w, 
            s=s[2], 
            dilation=self.dilation[2]
            )
        merge3 = tf.keras.layers.Concatenate(axis=-1)([merge2, conv3])

        conv4 = self.__conv_block__(
            merge3, 
            self.filter[3], 
            w=w, 
            s=s[3], 
            dilation=self.dilation[3]
            )
        merge4 = tf.keras.layers.Concatenate(axis=-1)([merge3, conv4])

        return merge4
       
    def __build__(
        self
        ):

        def one_hot(x):
            return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape1)
        input2 = tf.keras.layers.Input(shape=self.shape1)
        input3 = tf.keras.layers.Input(shape=self.shape2)
        input4 = tf.keras.layers.Input(shape=self.shape2)
        input5 = tf.keras.layers.Input(shape=self.shape3)
        input6 = tf.keras.layers.Input(shape=self.shape3)
        input7 = tf.keras.layers.Input(shape=self.shape4)

        # convert sequences to one-hot
        input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        merge1 = tf.keras.layers.Concatenate(axis=-1)([input1_oh, input2_oh])
        merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        feat1 = self.__conv_net__(
            tf.expand_dims(merge1, axis=-1), 
            w=(20, 4), 
            s=[(10, 4), 1, 1, 1]
            )
        feat2 = self.__conv_net__(
            tf.expand_dims(merge2, axis=-1), 
            w=(3, 1), 
            s=[(1, 1), 1, 1, 1]
            )

        merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        conv = self.__conv_block__(merge4, k=self.dense, w=(4, 4), s=(4, 4))
        reshape = tf.keras.layers.Permute((2, 3, 1))(conv)
        pool = tf.keras.layers.GlobalAveragePooling2D()(reshape)

        merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3, input7])   # concatenate global features before classifier layer

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(merge5)
        out = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(dense)

        self.model = tf.keras.models.Model(
            inputs=[input1, input2, input3, input4, input5, input6, input7], 
            outputs=out
            ) 
        optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optim, 
            loss='binary_crossentropy', 
            metrics=['accuracy', 
            tf.keras.metrics.AUC(name='auc')]
            ) 

    def fit(
        self, 
        inp1,                                                                   # 1st sequence
        inp2,                                                                   # 2nd sequence
        inp3,                                                                   # 1st read
        inp4,                                                                   # 2nd read
        inp5,                                                                   # 1st distance
        inp6,                                                                   # 2nd distance
        inp7,                                                                   # global features
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

        self.model.fit(
            [inp1, inp2, inp3, inp4, inp5, inp6, inp7], 
            y, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate], 
            validation_data=validation_data, 
            verbose=2
            )

    def predict(
        self, 
        inp1, 
        inp2, 
        inp3, 
        inp4, 
        inp5, 
        inp6,
        inp7):

        return self.model.predict([inp1, inp2, inp3, inp4, inp5, inp6, inp7])

    def save_model(
        self, 
        model_name
        ):

        self.model.save(model_name)