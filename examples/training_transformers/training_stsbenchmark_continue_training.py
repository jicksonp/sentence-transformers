"""
This example loads the pre-trained SentenceTransformer model 'bert-base-nli-mean-tokens' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import argparse

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
from datetime import datetime

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#### /print debug information to stdout


def main(base_model_name: str):
    # Read the dataset
    # model_name = 'bert-base-nli-mean-tokens'
    model_name = base_model_name
    train_batch_size = 16
    num_epochs = 4
    model_save_path = 'output/training_stsbenchmark_continue_training-' + model_name + '-' + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark', normalize_scores=True)

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(model_name)

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read STSbenchmark train dataset")
    train_dataset = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read STSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    print('Fine tuning started')
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)
    print('Fine tuning completed')

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################
    print('Evaluate the model started')
    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    model.evaluate(evaluator)
    print('Evaluate the model completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--base_model_name', help='Base model name', default='bert-base-nli-mean-tokens')

    args = parser.parse_args()
    main(
        args.base_model_name,
    )
