'''
    tf 2.5 for unilm
'''

import tensorflow as tf


def create_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def gelu(x):
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_model_weights_from_checkpoint(checkpoint_file, model, num_hidden_layers):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    weights = [
        loader('bert/embeddings/position_embeddings'),
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/token_type_embeddings'),
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ]
    model.get_layer('embeddings').set_weights(weights)

    for i in range(num_hidden_layers):
        pre = 'bert/encoder/layer_' + str(i) + '/'
        weights = [
            loader(pre + 'attention/self/query/kernel'),
            loader(pre + 'attention/self/query/bias'),
            loader(pre + 'attention/self/key/kernel'),
            loader(pre + 'attention/self/key/bias'),
            loader(pre + 'attention/self/value/kernel'),
            loader(pre + 'attention/self/value/bias'),
            loader(pre + 'attention/output/dense/kernel'),
            loader(pre + 'attention/output/dense/bias'),
            loader(pre + 'attention/output/LayerNorm/gamma'),
            loader(pre + 'attention/output/LayerNorm/beta'),
        ]
        model.get_layer('attention-' + str(i)).set_weights(weights)

        weights = [
            loader(pre + 'intermediate/dense/kernel'),
            loader(pre + 'intermediate/dense/bias'),
            loader(pre + 'output/dense/kernel'),
            loader(pre + 'output/dense/bias'),
            loader(pre + 'output/LayerNorm/gamma'),
            loader(pre + 'output/LayerNorm/beta'),
        ]
        model.get_layer('feedford-' + str(i)).set_weights(weights)

    weights = [
        loader('cls/predictions/output_bias'),
        loader('cls/predictions/transform/dense/kernel'),
        loader('cls/predictions/transform/dense/bias'),
        loader('cls/predictions/transform/LayerNorm/gamma'),
        loader('cls/predictions/transform/LayerNorm/beta')
    ]
    model.get_layer('Sequence').set_weights(weights)


class UniLMMask(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, **kwargs):
        super(UniLMMask, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

    def get_config(self):
        config = {
            'num_attention_heads': self.num_attention_heads,
        }
        base_config = super(UniLMMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        sen, seg = inputs
        batch_size = tf.shape(sen)[0]
        max_ls = tf.shape(sen)[1]
        sequence_mask = tf.greater(sen, 0)
        mask1 = tf.logical_and(sequence_mask, tf.equal(seg, 0))
        one_mask = tf.tile(tf.expand_dims(mask1, 1), [self.num_attention_heads, max_ls, 1])

        mask2 = tf.equal(seg, 1)
        two_mask = tf.tile(tf.expand_dims(mask2, 1), [self.num_attention_heads, max_ls, 1])
        future_mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(tf.range(0, limit=max_ls), max_ls), 0),
            [batch_size, 1, 1])
        future_mask = tf.tile(future_mask, [self.num_attention_heads, 1, 1])
        two_mask = tf.logical_and(two_mask, future_mask)

        mask_final = tf.logical_or(one_mask, two_mask)
        return mask_final


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, type_vocab_size, max_position_embeddings, hidden_size, hidden_dropout_prob,
                 **kwargs):
        super(Embeddings, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'type_vocab_size': self.type_vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_size': self.hidden_size,
            'hidden_dropout_prob': self.hidden_dropout_prob,
        }
        base_config = super(Embeddings, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.word_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size,
                                                         embeddings_initializer=create_initializer(),
                                                         dtype=tf.float32,
                                                         mask_zero=True,
                                                         name='word_embeddings')
        self.token_embeddings = tf.keras.layers.Embedding(self.type_vocab_size, self.hidden_size,
                                                          embeddings_initializer=create_initializer(),
                                                          dtype=tf.float32,
                                                          name='token_type_embeddings')
        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[self.max_position_embeddings, self.hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())
        self.layernorm = LayerNormalize(self.hidden_size, name='layernorm-pre')
        self.dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen, token_type_ids = inputs
        sen_embed = self.word_embeddings(sen)
        token_embed = self.token_embeddings(token_type_ids)
        seq_length = tf.shape(sen)[1]
        return self.dropout(self.layernorm(
            sen_embed + token_embed + self.position_embeddings[:seq_length])), self.word_embeddings.weights[0]


class LayerNormalize(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(LayerNormalize, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
        }
        base_config = super(LayerNormalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.gamma = self.add_weight('gamma', [self.hidden_size], tf.float32, tf.keras.initializers.Ones())
        self.beta = self.add_weight('beta', [self.hidden_size], tf.float32, tf.keras.initializers.Zeros())
        super(LayerNormalize, self).build(input_shape)

    @staticmethod
    def layer_norm(x, gamma, beta, epsilon=1.0e-6):
        """
        Layer norm raw computation.
        """
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * gamma + beta

    def call(self, x, **kwargs):
        return self.layer_norm(x, self.gamma, self.beta)


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.dense_q = tf.keras.layers.Dense(self.hidden_size,
                                             name='query',
                                             dtype=tf.float32,
                                             kernel_initializer=create_initializer())
        self.dense_k = tf.keras.layers.Dense(self.hidden_size,
                                             name='key',
                                             dtype=tf.float32,
                                             kernel_initializer=create_initializer())
        self.dense_v = tf.keras.layers.Dense(self.hidden_size,
                                             name='value',
                                             dtype=tf.float32,
                                             kernel_initializer=create_initializer())
        self.dense_o = tf.keras.layers.Dense(self.hidden_size,
                                             name='output',
                                             dtype=tf.float32,
                                             kernel_initializer=create_initializer())
        self.dropout1 = tf.keras.layers.Dropout(rate=self.attention_probs_dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.attention_probs_dropout_prob)
        self.layernorm = LayerNormalize(self.hidden_size, name='layernormattn')

        super(Attention, self).build(input_shape)

    def softmax(self, a, mask):
        """
        :param a: B*ML1*ML2
        :param mask: B*ML1*ML2
        """
        return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        q = tf.concat(tf.split(self.dense_q(x), self.num_attention_heads, axis=-1), axis=0)
        k = tf.concat(tf.split(self.dense_k(x), self.num_attention_heads, axis=-1), axis=0)
        v = tf.concat(tf.split(self.dense_v(x), self.num_attention_heads, axis=-1), axis=0)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(self.hidden_size / self.num_attention_heads)
        attention_output = self.dense_o(tf.concat(
            tf.split(tf.matmul(self.dropout1(self.softmax(qk, mask)), v), self.num_attention_heads, axis=0),
            axis=-1))
        return self.layernorm(x + self.dropout2(attention_output))


class FeedFord(tf.keras.layers.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob, **kwargs):
        super(FeedFord, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'hidden_dropout_prob': self.hidden_dropout_prob,
        }
        base_config = super(FeedFord, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.dense_ffgelu = tf.keras.layers.Dense(self.intermediate_size,
                                                  kernel_initializer=create_initializer(),
                                                  dtype=tf.float32,
                                                  name='intermediate',
                                                  activation=gelu)
        self.dense_ff = tf.keras.layers.Dense(self.hidden_size,
                                              kernel_initializer=create_initializer(),
                                              dtype=tf.float32,
                                              name='output')
        self.dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        self.layernorm = LayerNormalize(self.hidden_size, name='layernormffd')

        super(FeedFord, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.layernorm(x + self.dropout(self.dense_ff(self.dense_ffgelu(x))))


class Sequence(tf.keras.layers.Layer):
    def __init__(self, hidden_size, vocab_size, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
        }
        base_config = super(Sequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.transformer = tf.keras.layers.Dense(self.hidden_size,
                                                 activation=gelu,
                                                 kernel_initializer=create_initializer(),
                                                 dtype=tf.float32,
                                                 name='transformer')
        self.layernorm = LayerNormalize(self.hidden_size, name='layernormsuf')
        self.output_bias = self.add_weight('output_bias', [self.vocab_size], dtype=tf.float32,
                                           initializer=create_initializer())
        super(Sequence, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input, word_embeddings = inputs
        output = self.layernorm(self.transformer(input))
        output = tf.einsum('ijk,lk->ijl', output, word_embeddings)
        return output + self.output_bias


class CustomLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs):
        y_true, y_mask, y_pred, y_mask_sum = inputs
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = tf.reduce_sum(loss * y_mask) / y_mask_sum
        return loss

    def vae_acc(self, inputs):
        y_true, y_mask, y_pred, y_mask_sum = inputs
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = tf.reduce_sum(acc * y_mask) / y_mask_sum
        return acc

    def call(self, inputs, **kwargs):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = tf.cast(y_mask[:, 1:], tf.float32)  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        y_mask_sum = tf.reduce_sum(y_mask)

        loss = self.vae_loss((y_true, y_mask, y_pred, y_mask_sum))
        self.add_loss(loss)
        acc = self.vae_acc((y_true, y_mask, y_pred, y_mask_sum))
        self.add_metric(acc, aggregation="mean", name="acc")
        output = tf.multiply(tf.ones_like(inputs[2]), inputs[2], name='predict')
        return output


def build_model(vocab_size, type_vocab_size, max_position_embeddings, num_hidden_layers, hidden_size,
                intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
    sen = tf.keras.layers.Input(shape=[None, ], name='sen', dtype=tf.int32)
    token_type = tf.keras.layers.Input(shape=[None, ], name='token_type', dtype=tf.int32)

    mask = UniLMMask(num_attention_heads, name='mask')(inputs=(sen, token_type))
    now, weights = Embeddings(vocab_size, type_vocab_size, max_position_embeddings, hidden_size, hidden_dropout_prob,
                              name='embeddings')(inputs=(sen, token_type))
    for layers in range(num_hidden_layers):
        now = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                        name='attention-' + str(layers))(inputs=(now, mask))
        now = FeedFord(hidden_size, intermediate_size, hidden_dropout_prob, name='feedford-' + str(layers))(now)

    sequence = Sequence(hidden_size, vocab_size, name='Sequence')(inputs=(now, weights))
    output = CustomLossLayer(name='Losslayer')(inputs=(sen, token_type, sequence))
    return tf.keras.models.Model(inputs=[sen, token_type], outputs=[output])


if __name__ == "__main__":
    model = build_model(21128, 2, 512, 12, 768, 768 * 4, 12, 0.1, 0.1)
    # model_variables = model.outputs
    # for vi in model_variables:
    #     print(vi.name)

    load_model_weights_from_checkpoint("../chatbot/pretrained/chinese_L-12_H-768_A-12/bert_model.ckpt", model, 12)
    model.summary(line_length=180, positions=[0.2, 0.6, 0.8, 1.0])
    model.save("D:/bert.h5")

    # model = build_model(12, 20, 21128, 2, 512, 12, 768, 768 * 4, 12, 0.1, 0.1, 1e-5)
    # model.load_weights("D:/bert.h5")
    # model.summary()
    # model = tf.keras.models.load_model("D:/bert.h5",
    #                                    custom_objects={
    #                                        "Embeddings": Embeddings,
    #                                        "Mask": Mask,
    #                                        "Attention": Attention,
    #                                        "FeedFord": FeedFord,
    #                                        "SplitSequence": SplitSequence,
    #                                        "SplitPooler": SplitPooler,
    #                                        "Sequence": Sequence,
    #                                        "Pooler": Pooler,
    #                                        "NER": NER,
    #                                        "Classify": Classify,
    #                                        "ner_loss": ner_loss,
    #                                        "ner_accuracy": ner_accuracy,
    #                                    })
    # model.summary()
