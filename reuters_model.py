from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Multiply, multiply
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU 
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.engine.topology import Layer
from keras.optimizers import SGD

from keras import backend as K
from keras import regularizers,initializers


class AttentionLayer(Layer):
    '''
    Attention layer. 
    '''
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        ''' 
        具体来定义权重是怎么样的
        input_shape: (None, 80, 200)
        '''
        input_dim = input_shape[-1] # 200
        self.Uw = self.add_weight(name='Uw', shape=((input_dim, 1)), initializer='glorot_uniform',
                                  trainable=True)
        # 可训练的权应该在这里被加入列表self.trainable_weights中
        self.trainable_weights = [self.Uw]
        super(AttentionLayer, self).build(input_shape)


    def compute_mask(self, input, mask):
        return mask


    def call(self, x, mask=None):
        ''' 
        核心部分，定义向量是如何运算的
        x shape: (?, ?, 200)
        '''
        #print(K.int_shape(x))   # (None, 80, 200)
        #print(K.int_shape(self.Uw)) # (200, 1)
        multData =  K.exp(K.dot(x, self.Uw)) # shape: (None, 80, 1)
        # epsilon : 以数值形式返回一个（一般来说很小的）数，用以防止除0错误
        output = multData/(K.sum(multData, axis=1)+K.epsilon())[:,None]
        #print(K.int_shape(output))  #(None, 80, 1)
        return output


    def compute_output_shape(self, input_shape): # input_shape : (None, 80, 200)
        newShape = list(input_shape)
        newShape[-1] = 1
        return tuple(newShape)


def createHierarchicalAttentionModel(maxSeq, embeddingSize=None, vocabSize=None,
                        recursiveClass=GRU, wordRnnSize=100, sentenceRnnSize=100,
                        dropWordEmb=0.2, dropWordRnnOut=0.2, dropSentenceRnnOut=0.5):

    ## Sentence level logic -------------------------------------------------------------------------------- 
    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input') # Shape: (None, 80)
    emb = Embedding(vocabSize, embeddingSize)(wordsInputs) # shape: (None, 80, 200)

    if dropWordEmb != 0.0:
        emb = Dropout(dropWordEmb)(emb) # shape: (None, 80, 200)

    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(emb) # shape: (None, 80, 200)
    if dropWordRnnOut  > 0.0:
        wordRnn = Dropout(dropWordRnnOut)(wordRnn) # shape: (None, 80, 200)
    # Attention层会把权重alpha计算出来，根据权重计算的代码在后面
    attention = AttentionLayer()(wordRnn) # shape: (None, 80, 200)

    sentenceEmb = Lambda(lambda x:x[1]*x[0], output_shape=lambda x:x[0])([wordRnn, attention]) # shape: (None, 80, 200)
    sentenceEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(sentenceEmb) # shape: (None, 200)

    modelSentence = Model(inputs=wordsInputs, outputs=sentenceEmb)
    print(modelSentence.summary())

    ## Sentence level logic -----------------------------------------------------------------------------end

    ## Document level logic -------------------------------------------------------------------------------- 
    documentInputs = Input(shape=(None, maxSeq), dtype='int32', name='document_input') # shape: (None, None, 80)
    sentenceEmbbeding = TimeDistributed(modelSentence)(documentInputs) # shape: (None, None, 200)
    sentenceRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(sentenceEmbbeding) # shape: (None, None, 200)

    if dropSentenceRnnOut > 0.0:
        sentenceRnn = Dropout(dropSentenceRnnOut)(sentenceRnn) # shape: (None, None, 200)
    attentionSent = AttentionLayer()(sentenceRnn) # shape: (None, None, 200)

    documentEmb = multiply(inputs=[sentenceRnn, attentionSent]) # shape: (None, None, 200)
    # documentEmb = Merge([sentenceRnn, attentionSent], mode=lambda x:x[1]*x[0], output_shape=lambda x:x[0])
    documentEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]), name="att2")(documentEmb) # shape: (None, 200)
    documentOut = Dense(46, activation="softmax", name="documentOut")(documentEmb)

    ## Document level logic -----------------------------------------------------------------------------end 

    model = Model(input=[documentInputs], output=[documentOut])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    print(model.summary())
    return model

