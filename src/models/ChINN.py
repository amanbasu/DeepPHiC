import tensorflow as tf
import xgboost as xgb

'''
TensorFlow implementation of ChINN (https://github.com/mjflab/chinn)
'''

class ChINN:
    def __init__(
        self,
        filter=[128, 256, 128],
        dropout=0.5,
        dense=1024,
        shape1=(2000),                                                          # sequence features
        shape2=(1),                                                             # distance features
        learning_rate=1e-3
        ):
        
        self.filter = filter
        self.dropout = dropout
        self.dense = dense
        self.shape1 = shape1
        self.shape2 = shape2
        self.learning_rate = learning_rate

        self.model = None                                                       # TensorFlow model
        self.bst = None                                                         # XGBoost classifier
        self.feature_extractor = None                                           # TensorFlow model w/o classifier layer

        self.leaky = tf.keras.layers.LeakyReLU(alpha=1/5.5)
        
        tf.random.set_seed(0)
        self.__build__()
    
    def __feature_extractor__(
        self, 
        x, 
        k=11,                                                                   # kernel size
        s=2,                                                                    # stride size
        d=56
        ):

        conv1 = tf.keras.layers.Conv1D(
            self.filter[0], 
            k,
            s,
            activation=self.leaky
            )(x)
        pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv1)
        drop1 = tf.keras.layers.Dropout(self.dropout)(pool1)

        conv2 = tf.keras.layers.Conv1D(
            self.filter[1], 
            k, 
            s, 
            activation=self.leaky
            )(drop1)
        pool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv2)
        drop2 = tf.keras.layers.Dropout(self.dropout)(pool2)

        conv3 = tf.keras.layers.Conv1D(
            self.filter[2], 
            k, 
            s, 
            activation=self.leaky
            )(drop2)
        drop3 = tf.keras.layers.Dropout(self.dropout)(conv3)

        weight_sum = tf.keras.layers.Dense(
            d, 
            kernel_initializer=tf.keras.initializers.HeNormal
            )(drop3)
        weight_sum = tf.math.reduce_sum(weight_sum, axis=2)
        weight_sum = tf.keras.activations.tanh(weight_sum)

        return weight_sum

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

        feat1 = self.__feature_extractor__(input1_oh)
        feat2 = self.__feature_extractor__(input2_oh)

        merge = tf.keras.layers.Concatenate(axis=1)(
            [feat1, feat2, input3, input4]
            )
    
        dense = tf.keras.layers.Dense(
            self.dense, 
            activation=self.leaky, 
            kernel_initializer=tf.keras.initializers.HeNormal
            )(merge)
        out = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=tf.keras.initializers.HeNormal
            )(dense)

        self.model = tf.keras.models.Model(
            inputs=[input1, input2, input3, input4], 
            outputs=out
            ) 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=opt, 
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            ) 

    '''
    ChINN only uses sequence and distance features. 
    Hence, inp3 and inp4 are ignored.
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

        # ignore the read features
        validation_data = (
            validation_data[0][:2] + validation_data[0][4:],
            validation_data[1]
            )

        self.model.fit(
            [inp1, inp2, inp5, inp6], 
            y, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stop, learning_rate], 
            validation_data=validation_data, 
            verbose=2
            )

        ############# gradient boosting #############
        self.feature_extractor = tf.keras.models.Model(
            inputs=self.model.input, 
            outputs=self.model.layers[-3].output
            ) 

        features_train = self.feature_extractor.predict([inp1, inp2, inp5, inp6])
        features_val = self.feature_extractor.predict(validation_data[0])

        dtrain = xgb.DMatrix(features_train, label=y)
        dval = xgb.DMatrix(features_val, label=validation_data[1])
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        params = {
            'max_depth': 6, 
            'objective': 'binary:logistic', 
            'max_delta_step':1,
            'eta': 0.1, 
            'nthread': 4, 
            'eval_metric': ['aucpr', 'map'], 
            'seed': 0}
        self.bst = xgb.train(
            params, 
            dtrain, 
            1000, 
            evallist, 
            early_stopping_rounds=40,
            verbose_eval=True
            )

    '''
    ChINN only uses sequence and distance features. 
    Hence, inp3 and inp4 are ignored.
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

        features_test = self.feature_extractor.predict([inp1, inp2, inp5, inp6])
        dtest = xgb.DMatrix(features_test)

        return self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)

    def save_model(
        self, 
        model_name
        ):

        self.model.save(model_name)