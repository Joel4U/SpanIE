import argparse
from src.config import Config
import time
from src.model import TransformersCRF
import torch
from typing import List
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
from tqdm import tqdm
from src.data import TransformersNERREDataset
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:3", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="scierc")
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=20, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Usually we set to 100.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=80, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")
    parser.add_argument('--fp16', type=int, choices=[0, 1], default=0, help="use 16-bit floating point precision instead of 32-bit")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden size of the Span and Span pair")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")

    parser.add_argument('--embedder_type', type=str, default="roberta-large", help="you can use 'bert-base-uncased' and so on")
    parser.add_argument('--add_iobes_constraint', type=int, default=0, choices=[0,1], help="add IOBES constraint for transition parameters to enforce valid transitions")

    parser.add_argument("--print_detail_f1", type= int, default= 0, choices= [0, 1], help= "Open and close printing f1 scores for each tag after each evaluation epoch")
    parser.add_argument("--earlystop_atr", type=str, default="micro", choices= ["micro", "macro"], help= "Choose between macro f1 score and micro f1 score for early stopping evaluation")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="training model or test mode")
    parser.add_argument('--test_file', type=str, default="data/conll2003_sample/test.txt", help="test file for test mode, only applicable in test mode")

    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
    ### Data Processing Info
    train_num = len(train_loader)
    logger.info(f"[Data Info] number of training instances: {train_num}")

    logger.info(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}")
    logger.info(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.")
    logger.info(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.")
    model = TransformersCRF(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(model=model, learning_rate=config.learning_rate,
                                                                   num_training_steps=len(train_loader) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
    logger.info(f"[Optimizer Info] Modify the optimizer info as you need.")
    logger.info(optimizer)

    model.to(config.device)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    no_incre_dev = 0
    logger.info(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs")
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        for iter, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss = model(subword_input_ids=batch.input_ids.to(config.device), orig_to_tok_index=batch.orig_to_tok_index.to(config.device),
                             attention_mask=batch.attention_mask.to(config.device), word_seq_lens=batch.word_seq_len.to(config.device),
                             span_ids=batch.span_ids.to(config.device), span_mask=batch.span_mask.to(config.device),
                             span_ner_labels=batch.span_ner_labels.to(config.device), relation_indices=batch.relation_indices.to(config.device),
                             relation_labels=batch.relation_labels.to(config.device), relation_mask=batch.relation_mask.to(config.device))
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        logger.info(f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        model.metric_reset()
        dev_f1 = evaluate_model(config, model, dev_loader, "dev")
        model.metric_reset()
        test_f1 = evaluate_model(config, model, test_loader, "test")
        if dev_f1 > best_dev[0]:
            # logger.info(f"saving the best model with best dev f1 score {dev_metrics[2]}")
            no_incre_dev = 0
            best_dev[0] = dev_f1
            best_dev[1] = i
            best_test[0] =test_f1
            best_test[1] = i
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            logger.info("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break


def evaluate_model(config: Config, model: TransformersCRF, data_loader: DataLoader, name: str, print_each_type_metric: bool = False):
    ## evaluation
    total_ner_dict = {'p': 0, 'r': 0, 'f': 0}
    total_rel_dict = {'p': 0, 'r': 0, 'f': 0}
    total_batches = len(data_loader)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config.fp16)):
        for batch_id, batch in enumerate(data_loader, 0):
            span_metrics,  rel_metrics = model(subword_input_ids=batch.input_ids.to(config.device),
                         orig_to_tok_index=batch.orig_to_tok_index.to(config.device),
                         attention_mask=batch.attention_mask.to(config.device), word_seq_lens=batch.word_seq_len.to(config.device),
                         span_ids=batch.span_ids.to(config.device), span_mask=batch.span_mask.to(config.device),
                         span_ner_labels=batch.span_ner_labels.to(config.device), relation_indices=batch.relation_indices.to(config.device),
                         relation_labels=batch.relation_labels.to(config.device), relation_mask=batch.relation_mask.to(config.device), is_train= False)
            ner_prf = span_metrics.get_metric()
            rel_prf = rel_metrics.get_metric()

            total_ner_dict['p'] += ner_prf['p']
            total_ner_dict['r'] += ner_prf['r']
            total_ner_dict['f'] += ner_prf['f']
            total_rel_dict['p'] += rel_prf['p']
            total_rel_dict['r'] += rel_prf['r']
            total_rel_dict['f'] += rel_prf['f']
            batch_id += 1

    # 计算除最后一个批次之外的所有批次的平均值
    avg_ner_p = total_ner_dict['p'] / (total_batches - 1)
    avg_ner_r = total_ner_dict['r'] / (total_batches - 1)
    avg_ner_f = total_ner_dict['f'] / (total_batches - 1)
    avg_rel_p = total_rel_dict['p'] / (total_batches - 1)
    avg_rel_r = total_rel_dict['r'] / (total_batches - 1)
    avg_rel_f = total_rel_dict['f'] / (total_batches - 1)

    # 单独计算最后一个批次的评估指标值
    last_batch_ner_p = ner_prf['p']
    last_batch_ner_r = ner_prf['r']
    last_batch_ner_f = ner_prf['f']
    last_batch_rel_p = rel_prf['p']
    last_batch_rel_r = rel_prf['r']
    last_batch_rel_f = rel_prf['f']

    # 计算平均值
    ner_p = (total_ner_dict['p'] + last_batch_ner_p) / total_batches * 100
    ner_r = (total_ner_dict['r'] + last_batch_ner_r) / total_batches * 100
    ner_f = (total_ner_dict['f'] + last_batch_ner_f) / total_batches * 100
    rel_p = (total_rel_dict['p'] + last_batch_rel_p) / total_batches * 100
    rel_r = (total_rel_dict['r'] + last_batch_rel_r) / total_batches * 100
    rel_f = (total_rel_dict['f'] + last_batch_rel_f) / total_batches * 100
    logger.info(f"[{name} set Total] NER Prec.: {ner_p:.2f}, Rec.: {ner_r:.2f}, Micro F1: {ner_f:.2f}\r")
    logger.info(f"[{name} set Total] REL Prec.: {rel_p:.2f}, Rec.: {rel_r:.2f}, Micro F1: {rel_f:.2f}")
    return [ner_p, ner_r, ner_f]


def main():
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=True)
    logger.info(f"[Data Info] Reading dataset from: \n{conf.train_file}\n{conf.dev_file}\n{conf.test_file}")
    train_dataset = TransformersNERREDataset(conf.train_file, tokenizer, is_train=True)
    conf.ner_label = train_dataset.ner_label
    conf.rel_label = train_dataset.re_label
    dev_dataset = TransformersNERREDataset(conf.dev_file, tokenizer, ner_label=train_dataset.ner_label, re_label=train_dataset.re_label, is_train=False)
    test_dataset = TransformersNERREDataset(conf.test_file, tokenizer, ner_label=train_dataset.ner_label, re_label=train_dataset.re_label, is_train=False)
    num_workers = 0
    conf.label_size = train_dataset.ner_label.label_num
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=test_dataset.collate_fn)

    train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
