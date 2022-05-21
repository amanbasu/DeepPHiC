import tensorflow as tf
from DeepPHiC import DeepPHiC

'''
Seq + reads
'''
class DeepPHiCII(DeepPHiC):
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
        # merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

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

        # merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3])

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(pool)
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

'''
Seq + dist
'''
class DeepPHiCIII(DeepPHiC):
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
        # merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        feat1 = self.__conv_net__(
            tf.expand_dims(merge1, axis=-1), 
            w=(20, 4), 
            s=[(10, 4), 1, 1, 1]
            )
        # feat2 = self.__conv_net__(
        #     tf.expand_dims(merge2, axis=-1), 
        #     w=(3, 1), 
        #     s=[(1, 1), 1, 1, 1]
        #     )

        # merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        conv = self.__conv_block__(feat1, k=self.dense, w=(4, 4), s=(4, 4))
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

'''
Sequence only
'''
class DeepPHiCIV(DeepPHiC):
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
        # merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        # merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        feat1 = self.__conv_net__(
            tf.expand_dims(merge1, axis=-1), 
            w=(20, 4), 
            s=[(10, 4), 1, 1, 1]
            )
        # feat2 = self.__conv_net__(
        #     tf.expand_dims(merge2, axis=-1), 
        #     w=(3, 1), 
        #     s=[(1, 1), 1, 1, 1]
        #     )

        # merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        conv = self.__conv_block__(feat1, k=self.dense, w=(4, 4), s=(4, 4))
        reshape = tf.keras.layers.Permute((2, 3, 1))(conv)
        pool = tf.keras.layers.GlobalAveragePooling2D()(reshape)

        # merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3])

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(pool)
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

'''
Reads only
'''
class DeepPHiCV(DeepPHiC):
    def __build__(
        self
        ):

        # def one_hot(x):
        #     return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape1)
        input2 = tf.keras.layers.Input(shape=self.shape1)
        input3 = tf.keras.layers.Input(shape=self.shape2)
        input4 = tf.keras.layers.Input(shape=self.shape2)
        input5 = tf.keras.layers.Input(shape=self.shape3)
        input6 = tf.keras.layers.Input(shape=self.shape3)

        # convert sequences to one-hot
        # input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        # input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        # merge1 = tf.keras.layers.Concatenate(axis=-1)([input1_oh, input2_oh])
        merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        # merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        # feat1 = self.__conv_net__(
        #     tf.expand_dims(merge1, axis=-1), 
        #     w=(20, 4), 
        #     s=[(10, 4), 1, 1, 1]
        #     )
        feat2 = self.__conv_net__(
            tf.expand_dims(merge2, axis=-1), 
            w=(3, 1), 
            s=[(1, 1), 1, 1, 1]
            )

        # merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        conv = self.__conv_block__(feat2, k=self.dense, w=(4, 4), s=(4, 4))
        reshape = tf.keras.layers.Permute((2, 3, 1))(conv)
        pool = tf.keras.layers.GlobalAveragePooling2D()(reshape)

        # merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3])

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(pool)
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

'''
Distance only
'''
class DeepPHiCVI(DeepPHiC):
    def __build__(
        self
        ):

        # def one_hot(x):
        #     return tf.one_hot(tf.cast(x, 'uint8'), 4) 

        input1 = tf.keras.layers.Input(shape=self.shape1)
        input2 = tf.keras.layers.Input(shape=self.shape1)
        input3 = tf.keras.layers.Input(shape=self.shape2)
        input4 = tf.keras.layers.Input(shape=self.shape2)
        input5 = tf.keras.layers.Input(shape=self.shape3)
        input6 = tf.keras.layers.Input(shape=self.shape3)

        # convert sequences to one-hot
        # input1_oh = tf.keras.layers.Lambda(one_hot)(input1)
        # input2_oh = tf.keras.layers.Lambda(one_hot)(input2)

        # merge1 = tf.keras.layers.Concatenate(axis=-1)([input1_oh, input2_oh])
        # merge2 = tf.keras.layers.Concatenate(axis=-1)([input3, input4])
        merge3 = tf.keras.layers.Concatenate(axis=-1)([input5, input6])

        # feat1 = self.__conv_net__(
        #     tf.expand_dims(merge1, axis=-1), 
        #     w=(20, 4), 
        #     s=[(10, 4), 1, 1, 1]
        #     )
        # feat2 = self.__conv_net__(
        #     tf.expand_dims(merge2, axis=-1), 
        #     w=(3, 1), 
        #     s=[(1, 1), 1, 1, 1]
        #     )

        # merge4 = tf.keras.layers.Concatenate(axis=1)([feat1, feat2])
        # conv = self.__conv_block__(merge4, k=self.dense, w=(4, 4), s=(4, 4))
        # reshape = tf.keras.layers.Permute((2, 3, 1))(conv)
        # pool = tf.keras.layers.GlobalAveragePooling2D()(reshape)

        # merge5 = tf.keras.layers.Concatenate(axis=-1)([pool, merge3])

        dense = tf.keras.layers.Dense(
            self.dense, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0)
            )(merge3)
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