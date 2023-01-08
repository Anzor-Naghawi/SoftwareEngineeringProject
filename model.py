from data import Data, tb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


class Model:
    def __init__(self):
        self.data = Data(fn='PDFMalware2022.parquet')
        self.x = None
        self.y = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.scaler = None
        self.xtrainsc = None
        self.xtestsc = None
        self.logmdl = None
        self.testvals = None
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        self.model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=0.03),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        self.fittedmodel = None

        self.treemodel = DecisionTreeRegressor()
        self.treemodeltest = DecisionTreeClassifier()

    def split(self):
        self.x = self.data.dataframe.drop('Class', axis=1)
        self.x = self.x.drop('FileName', axis=1)

        self.x[['Images',
                'Text',
                'Header',
                'Obj',
                'Endobj',
                'Stream',
                'Endstream',
                'Xref',
                'StartXref',
                'PageNo',
                'JS',
                'Javascript',
                'AA',
                'OpenAction',
                'Acroform',
                'JBIG2Decode',
                'RichMedia',
                'Launch',
                'EmbeddedFile',
                'XFA'
                ]] = self.x[['Images',
                             'Text',
                             'Header',
                             'Obj',
                             'Endobj',
                             'Stream',
                             'Endstream',
                             'Xref',
                             'StartXref',
                             'PageNo',
                             'JS',
                             'Javascript',
                             'AA',
                             'OpenAction',
                             'Acroform',
                             'JBIG2Decode',
                             'RichMedia',
                             'Launch',
                             'EmbeddedFile',
                             'XFA'
                             ]].apply(lambda col: pd.Categorical(col).codes)
        print(self.x.dtypes)
        self.x = self.x.astype(float)
        self.y = self.data.dataframe['Class']
        self.y = self.y.replace(['Malicious'], [1])
        self.y = self.y.replace(['Benign'], [0])
        for x in self.y:
            print(x)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.x, self.y,
            test_size=0.2, random_state=42
        )
        print(self.xtrain)

    def scale(self):
        self.scaler = StandardScaler()
        print(self.xtrain)
        self.xtrainsc = self.scaler.fit_transform(self.xtrain)
        self.xtestsc = self.scaler.transform(self.xtest)
        print(self.xtrainsc)
        self.fittedmodel = self.model.fit(self.xtrainsc, self.ytrain, epochs=1)

    def fitmodel2(self):
        self.treemodel.fit(self.xtrain, self.ytrain)

    def test2(self):
        outputs = self.treemodel.predict(self.xtest)
        crtprd = 0
        for j in range(len(self.ytest.tolist())):
            if self.ytest.tolist()[j] == outputs.tolist()[j]:
                crtprd += 1

        print("MODEL 2 Accuracy is ", crtprd/len(self.ytest))

    def test(self):
        self.testvals = self.model.evaluate(self.xtestsc, self.ytest)

    def plots(self):
        rcParams['figure.figsize'] = (18, 8)
        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        plt.plot(
            np.arange(1, 101),
            self.fittedmodel.history['loss'], label='Loss'
        )
        plt.show()
        plt.plot(
            np.arange(1, 101),
            self.fittedmodel.history['accuracy'], label='Accuracy'
        )
        plt.show()
        plt.plot(
            np.arange(1, 101),
            self.fittedmodel.history['precision'], label='Precision'
        )
        plt.show()
        plt.plot(
            np.arange(1, 101),
            self.fittedmodel.history['recall'], label='Recall'
        )
        plt.show()
        plt.title('Evaluation metrics', size=20)
        plt.xlabel('Epoch', size=14)
        plt.legend()