"""
    textsum using unilm
"""

import tensorflow as tf
import os
from utils import single_example_parser, batched_data, checksentence, replace
from unilm import build_model, load_model_weights_from_checkpoint
from tokenizers import BertWordPieceTokenizer
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--check', type=str, default='model/textsum/', help='The path where model shall be saved')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--steps_per_epoch', type=int, default=10000, help='Epochs during training')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learing rate')
parser.add_argument('--mode', type=str, default='train1', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict'
                                                               'makepb: make pb file')
params = parser.parse_args()

tokenizer = BertWordPieceTokenizer("pretrained/chinese_L-12_H-768_A-12/vocab.txt")
fw = open("result/textsum.log", "a+", encoding="utf-8")


class CheckCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(params.check + 'textsum.h5')

        sentences = [
            "2007年乔布斯向人们展示iPhone并宣称“它将会改变世界”，还有人认为他在夸大其词，然而在8年后，以iPhone为代表的触屏智能手机已经席卷全球各个角落。未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更大的贡献。",
            "目前世界上有着几百种编程语言，我应该学哪个?如何选择“正确”的编程语言进行学习?我所学的语言日后能否成为我获取好生活的保障?在这个问题上，很多人都曾经给出了他们都看法。但在我看来，这个问题答案其实非常简单：那就是JavaScript。"
        ]
        for sentence in sentences:
            result = checksentence(replace(sentence).lower().strip(), self.model, tokenizer)
            fw.write("原文：" + sentence + "\n摘要：" + result + "\n\n")


class TextSum():
    def build_model(self):
        model = build_model(21128, 2, 512, 12, 768, 768 * 4, 12, 0.1, 0.1)

        return model

    def train(self, train_file):
        model = self.build_model()
        if params.mode == 'train0':
            load_model_weights_from_checkpoint("pretrained/chinese_L-12_H-768_A-12/bert_model.ckpt",
                                               model, 12)
            model.summary(line_length=170, positions=[0.2, 0.6, 0.8, 1.0])

            if not os.path.exists(params.check):
                os.makedirs(params.check)
            model.save_weights(params.check + "textsum.h5")

        else:
            model.load_weights(params.check + "textsum.h5")

        train_batch = batched_data(train_file,
                                   single_example_parser,
                                   params.batch_size,
                                   padded_shapes=(([-1], [-1]), [-1]))

        model.compile(tf.keras.optimizers.Adam(params.lr))
        model.fit(train_batch,
                  epochs=params.epochs,
                  steps_per_epoch=params.steps_per_epoch,
                  callbacks=[CheckCallback()]
                  )
        fw.close()

    def predict(self, sentences):
        model = self.build_model()
        model.load_weights(params.check + "textsum.h5")
        for sentence in sentences:
            sentence = replace(sentence.lower().strip())
            result = checksentence(sentence, model, tokenizer)
            print("\n原文：" + sentence + "\n摘要：" + result)

    def h52pb(self):
        """ convert keras h5 model file to frozen graph(.pb file)
        """
        from tensorflow.python.framework import graph_io

        def freeze_graph(graph, session, output_node_names, model_name):
            with graph.as_default():
                graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
                graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
                graph_io.write_graph(graphdef_frozen, params.check, os.path.basename(model_name) + ".pb", as_text=False)

        tf.keras.backend.set_learning_phase(0)  # this line most important
        model = self.build_model()
        model.load_weights(params.check + "textsum.h5")
        session = tf.keras.backend.get_session()
        freeze_graph(session.graph, session, [out.op.name for out in model.outputs], "textsum")


if __name__ == '__main__':
    textsum = TextSum()
    if params.mode.startswith("train"):
        textsum.train([
            "data/TFRecordFiles/LCSTS_I.tfrecord"
        ])

    elif params.mode == 'predict':
        sentences = [
            "2007年乔布斯向人们展示iPhone并宣称“它将会改变世界”，还有人认为他在夸大其词，然而在8年后，以iPhone为代表的触屏智能手机已经席卷全球各个角落。未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更大的贡献。",
            "目前世界上有着几百种编程语言，我应该学哪个?如何选择“正确”的编程语言进行学习?我所学的语言日后能否成为我获取好生活的保障?在这个问题上，很多人都曾经给出了他们都看法。但在我看来，这个问题答案其实非常简单：那就是JavaScript。"
        ]
        textsum.predict(sentences)
    elif params.mode == "makepb":
        textsum.h52pb()
