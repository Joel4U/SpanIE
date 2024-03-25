# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizerFast, RobertaTokenizer, AutoTokenizer
import collections
import numpy as np
import itertools
from src.data.data_utils import build_label_idx, check_all_labels_in_dict, format_label_fields, LabelField
import json
import re
from src.data import Instance
from termcolor import colored
import warnings
from transformers.tokenization_utils_base import BatchEncoding


class TransformersNERREDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizerFast,
                 ner_label: LabelField = None,
                 re_label: LabelField = None,
                 is_train: bool = False,
                 context_width: int = 1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int( (context_width - 1) / 2)
        self.max_span_width = 10
        self.max_entity_length = 0
        self.entity_length_counts = {}
        self.total_entities = 0
        insts = self.read_from_json(file=file)
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            self.ner_label = LabelField()
            self.re_label = LabelField()
        else:
            # assert self.ner_label.label_num is not 0  ## for dev/test dataset we don't build label2idx
            self.ner_label = ner_label
            self.re_label = re_label
        self.insts_ids = self.text_to_instance_ids(insts, tokenizer)
        self.tokenizer = tokenizer

    def read_from_json(self, file: str)-> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        insts = []
        data = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))

        for record in tqdm(data):
            sentence_start = 0
            n_sentences = len(record["sentences"])
            # TODO(Ulme) Make it so that the
            record["sentence_groups"] = [[self._normalize_word(word) for sentence in
                                        record["sentences"][max(0, i - self.k):min(n_sentences, i + self.k + 1)] for word
                                        in sentence] for i in range(n_sentences)]
            record["sentence_start_index"] = [
                sum(len(line["sentences"][i - j - 1]) for j in range(min(self.k, i))) if i > 0 else 0 for i in
                range(n_sentences)]
            record["sentence_end_index"] = [record["sentence_start_index"][i] + len(record["sentences"][i]) for i in
                                          range(n_sentences)]
            for sentence_group_nr in range(len(record["sentence_groups"])):
                if len(record["sentence_groups"][sentence_group_nr]) > 400:
                    # record["sentence_groups"][sentence_group_nr] = line["sentences"][sentence_group_nr]
                    record["sentence_start_index"][sentence_group_nr] = 0
                    record["sentence_end_index"][sentence_group_nr] = len(record["sentences"][sentence_group_nr])
                    if len(record["sentence_groups"][sentence_group_nr]) > 400:
                        warnings.warn("Sentence with > 400 words; BERT may truncate.")

            zipped = zip(record["sentences"], record["ner"], record["relations"], record["sentence_groups"],
                         record["sentence_start_index"], record["sentence_end_index"])

            for sentence_num, (sentence, ner, relations, groups, start_ix, end_ix) in enumerate(zipped):

                ner_dict, relation_dict = format_label_fields(ner, relations, sentence_start)
                # if len(ner_dict) > 0: # 统计实体跨度长度用
                #     for (start, end), label in ner_dict.items():
                #         chunk_len = end - start + 1
                #         if self.max_span_width < chunk_len:
                #             self.max_span_width = chunk_len
                #         self.entity_length_counts[chunk_len] = self.entity_length_counts.get(chunk_len, 0) + 1
                #         self.total_entities += len(ner_dict)
                sentence_start += len(sentence)
                words = [self._normalize_word(word) for word in sentence]
                insts.append([words, ner_dict, relation_dict])
                # sentence, spans, ner_labels, span_ner_labels, relation_indices, relation_labels = self.text_to_instance(
                #     sentence, tokenizer, ner_dict, relation_dict)
                #filter out sentences with only one entity.
                # if len(span_ner_labels) <= 0:
                #     continue
                # insts.append([sentence, spans, ner_labels, relation_indices, relation_labels])

        # print("number of sentences: {}".format(len(insts)))
        # print(colored(f"Entity Type Counts: {self.entity_length_counts}","yellow"))
        # # 打印不同实体跨度长度的实体个数占总数的百分比
        # print("Entity Length Percentages:")
        # for length, count in self.entity_length_counts.items():
        #     percentage = (count / self.total_entities) * 100
        #     print(f"Length {length}: {percentage:.2f}%\t")
        return insts

    def _normalize_word(self, word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def text_to_instance_ids(self, insts, tokenizer):
        features = []
        for idx, inst in tqdm(enumerate(insts)):
            words = [self._normalize_word(word) for word in inst[0]]
            orig_to_tok_index = []
            res = tokenizer.encode_plus(words, is_split_into_words=True) # RobertaTokenizerFast
            subword_idx2word_idx = res.word_ids(batch_index=0) # RobertaTokenizerFast
            prev_word_idx = -1
            for i, mapped_word_idx in enumerate(subword_idx2word_idx):
                if mapped_word_idx is None:  ## cls and sep token
                    continue
                if mapped_word_idx != prev_word_idx:
                    ## because we take the first subword to represent the whold word
                    orig_to_tok_index.append(i)
                    prev_word_idx = mapped_word_idx
            assert len(orig_to_tok_index) == len(words)

            spans = []
            span_ner_labels = set()
            ner_labels = []
            for start, end in self.enumerate_spans(inst[0], max_span_width=self.max_span_width):
                span_ix = (start, end)
                spans.append((start, end))
                ner_label = inst[1][span_ix] # inst[1] is ner_dict
                ner_labels.append(self.ner_label.get_id(ner_label))
                if ner_label:
                    span_ner_labels.add(span_ix)

            n_spans = len(spans)
            candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i != j]

            relation_indices = []
            relation_labels = []
            for i, j in candidate_indices:
                if spans[i] in span_ner_labels and spans[j] in span_ner_labels:
                    span_pair = (spans[i], spans[j])
                    relation_label = inst[2][span_pair] # inst[2] is relation_dict
                    if relation_label:
                        relation_indices.append((i, j))
                        relation_labels.append(self.re_label.get_id(relation_label))
            # Add negative re label
            self.re_label.get_id("")
            ##filter out sentences without the entity.
            if len(span_ner_labels) <= 1:
                continue
            features.append({"input_ids": res["input_ids"], "attention_mask": res["attention_mask"],
                             "orig_to_tok_index": orig_to_tok_index, "word_seq_len": len(orig_to_tok_index),
                             "all_span_ids": spans, "span_ner_label_ids": ner_labels,
                             "relation_indices": relation_indices, "relation_label_ids": relation_labels})
        return features


    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):

        max_span_width = max_span_width or len(sentence)
        spans = []

        for start_index in range(len(sentence)):
            last_end_index = min(start_index + max_span_width, len(sentence))
            first_end_index = min(start_index + min_span_width - 1, len(sentence))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                spans.append((start, end))
        return spans

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Dict]):
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])
        max_span_num = max([len(feature["span_ner_label_ids"]) for feature in batch])
        max_relation_num = max([len(feature["relation_label_ids"]) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            mask = feature["attention_mask"] + [0] * padding_length
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len

            padding_span_len = max_span_num - len(feature["span_ner_label_ids"])
            all_spans = feature["all_span_ids"] + [(0, 0)] * padding_span_len
            all_ner_labels = feature["span_ner_label_ids"] + [0] * padding_span_len
            all_span_mask = [0] * len(all_ner_labels)
            all_span_mask[:len(feature["span_ner_label_ids"])] = [1] * len(feature["span_ner_label_ids"])

            padding_relation_len = max_relation_num - len(feature["relation_label_ids"])
            all_relation_indices = feature["relation_indices"] + [(0, 0)] * padding_relation_len
            all_relation_labels = feature["relation_label_ids"] + [0] * padding_relation_len
            all_relation_mask = [0] * len(all_relation_labels)
            all_relation_mask[:len(feature["relation_label_ids"])] = [1] * len(feature["relation_label_ids"])

            batch[i] = {"input_ids": input_ids, "attention_mask": mask, "orig_to_tok_index": orig_to_tok_index, "word_seq_len": len(feature["orig_to_tok_index"]),
                        "span_ids": all_spans, "span_ner_labels": all_ner_labels, "span_mask": all_span_mask,
                        "relation_indices": all_relation_indices, "relation_labels": all_relation_labels, "relation_mask": all_relation_mask}
        encoded_inputs = {key: [example[key] for example in batch] for key in batch[0].keys()}
        results = BatchEncoding(encoded_inputs, tensor_type='pt')
        return results


## testing code to test the dataset
if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained('../../roberta-base/', add_prefix_space=True)
    dataset = TransformersNERREDataset(file="../../data/ace05/train.txt", tokenizer=tokenizer, is_train=True)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    print(len(train_dataloader))
    for batch in train_dataloader:
        print(batch.input_ids)
        pass
