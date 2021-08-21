import collections
import numpy as np
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from tensorflow.keras.models import Model
import re


def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        print(type(text))
        print('####################wrong################')


def load_vocab(vocab_file):  # 获取BERT字表方法
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert2Uni(tmp)
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'tok': tf.io.FixedLenSequenceFeature([], tf.int64)
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    tok = sequence_parsed['tok']
    return (sen, tok), sen


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .shuffle(buffer_size) \
        .repeat() \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset


def checksentence(sentence: str, model: Model, tokenizer: BertWordPieceTokenizer):
    tokenresult = tokenizer.encode(sentence)

    sen2id = tokenresult.ids
    tok2id = tokenresult.type_ids

    originlen = len(sen2id)

    endflag = False
    result = ""
    while not endflag:
        output = model.predict([np.array([sen2id]), np.array([tok2id])])[0, -1]
        resultid = np.argmax(output)
        tmpresult = tokenizer.decode([resultid])

        if tmpresult == "[SEP]":
            endflag = True
        elif len(sen2id) == originlen + 32:
            result += tmpresult
            endflag = True
        else:
            result += tmpresult
            sen2id = sen2id + [resultid]
            tok2id = tok2id + [1]

    return result


rep = {
    '“': '"',
    '”': '"',
}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))


def replace(words):
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], words)
