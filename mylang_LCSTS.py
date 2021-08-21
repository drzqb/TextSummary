import tensorflow as tf
from tqdm import tqdm
from utils import replace
from tokenizers import BertWordPieceTokenizer

maxword = 512


class Lang():
    def __init__(self):
        self.tokenizer = BertWordPieceTokenizer("pretrained/chinese_L-12_H-768_A-12/vocab.txt")

    def toid(self, sourcefile, tfrecordfile):
        m_samples = 0

        writer = tf.io.TFRecordWriter(tfrecordfile)
        max_len = 0
        lastline = ""

        with open(sourcefile, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = replace(line.lower().strip())

                if "<summary>" in lastline:
                    summary = line
                elif "<short_text>" in lastline:
                    sen = line
                    sen2id = self.tokenizer.encode(sen, summary)

                    if len(sen2id.type_ids) > maxword:
                        lastline = line
                        continue

                    # print(sen)
                    # print(summary + '\n')

                    lensen = len(sen2id.type_ids)
                    if lensen > max_len:
                        max_len = lensen

                    sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                                   sen2id.ids]
                    tok_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tok_])) for tok_ in
                                   sen2id.type_ids]

                    seq_example = tf.train.SequenceExample(
                        feature_lists=tf.train.FeatureLists(feature_list={
                            'sen': tf.train.FeatureList(feature=sen_feature),
                            'tok': tf.train.FeatureList(feature=tok_feature)
                        })
                    )

                    serialized = seq_example.SerializeToString()

                    writer.write(serialized)
                    m_samples += 1

                lastline = line

        print('\n')
        print('最大序列长度: {}'.format(max_len))  # III 152  II 156  I 163

        print('样本总量共：%d 句' % m_samples)  # III 1106句    II 10666句   I 2400591句


if __name__ == '__main__':
    lang = Lang()
    lang.toid(
        "E:/resources/LCSTS_ORIGIN/LCSTS_ORIGIN/DATA/PART_I.txt",
        "data/TFRecordFiles/LCSTS_I.tfrecord"
    )
