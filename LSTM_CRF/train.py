from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import argparse
import os


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--gpu', default=0, type=int)
    parser = args.parse_args()
    gpu = parser.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    main()

