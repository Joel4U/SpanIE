import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.config.eval import sequence_cross_entropy_with_logits
from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.embedder import TransformersEmbedder
from typing import Tuple, Union
from src.model.module.modules import EndpointSpanExtractor, SelfAttentiveSpanExtractor, FeedForward, \
                        flatten_and_batch_shift_indices, batched_index_select, SpanPairPairedLayer
from src.config.precision_recall_f1 import PrecisionRecallF1
from collections import defaultdict

class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.transformer = TransformersEmbedder(transformer_model_name=config.embedder_type)
        self.contextual_encoder = BiLSTMEncoder(input_dim=self.transformer.get_output_dim(), hidden_dim=config.hidden_dim, drop_lstm=config.dropout)
        # span_width_embedding_dim '64' is correlation to the self.span_pair_layer = SpanPairPairedLayer(dim_reduce_layer, repr_layer)
        # which default dist_emb_size is '64'
        self.endpoint_span_extractor = EndpointSpanExtractor(config.hidden_dim,
                                                              combination='x,y',
                                                              num_width_embeddings=32,
                                                              span_width_embedding_dim=64,
                                                              bucket_widths=True)
        self.attentive_span_extractor = SelfAttentiveSpanExtractor(self.transformer.get_output_dim())
        ## span predictioin layer
        input_dim = self.endpoint_span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim()
        self.span_layer = FeedForward(input_dim=input_dim, num_layers=2, hidden_dim=config.hidden_dim, dropout=0.3)
        self.span_proj_label = nn.Linear(config.hidden_dim, config.label_size)

        self.spans_per_word = 0.6 ## thershold number of spans in each sentence
        self.ner_neg_id = config.ner_label.get_id('')

        self.re_neg_id = config.rel_label.get_id('')
        self.e2e = True #End2End: if use gold relation index when training
        ## span pair
        re_label_num = config.rel_label.label_num
        dim_reduce_layer = FeedForward(input_dim, num_layers=1, hidden_dim=config.hidden_dim)
        repr_layer = FeedForward(config.hidden_dim * 3 + 64, num_layers=2, hidden_dim=config.hidden_dim//4)
        self.span_pair_layer = SpanPairPairedLayer(dim_reduce_layer, repr_layer)
        self.span_pair_label_proj = nn.Linear(config.hidden_dim//4, re_label_num)

        ## metrics
        # ner
        self.ner_prf = PrecisionRecallF1(neg_label=self.ner_neg_id)
        # relation
        self.re_prf = PrecisionRecallF1(neg_label=self.re_neg_id)

    def forward(self, subword_input_ids: torch.Tensor, attention_mask: torch.Tensor, orig_to_tok_index: torch.Tensor, word_seq_lens: torch.Tensor,
                    span_ids: torch.Tensor, span_mask: torch.Tensor, relation_indices: torch.Tensor, relation_mask: torch.Tensor,
                    span_ner_labels: torch.Tensor = None, relation_labels: torch.Tensor = None,
                    is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bz, seq_len = orig_to_tok_index.size()
        word_rep = self.transformer(subword_input_ids, orig_to_tok_index, attention_mask)
        contextual_encoding = self.contextual_encoder(word_rep, word_seq_lens.cpu())
        # extract span representation
        ep_span_emb = self.endpoint_span_extractor(contextual_encoding, span_ids.long(), span_mask)  # (batch_size , span_num, hidden_size)
        att_span_emb = self.attentive_span_extractor(word_rep, span_ids.long(), span_mask)  # (batch_size, span_num, bert_dim)
        span_emb = torch.cat((ep_span_emb, att_span_emb), dim=-1)  # (batch_size, span_num, hidden_size+bert_dim)

        span_logits = self.span_proj_label(self.span_layer(span_emb))  # (batch_size, span_num, span_label_num)
        span_prob = F.softmax(span_logits, dim=-1)  # (batch_size, span_num, span_label_num)
        _, span_pred = span_prob.max(2)

        span_prob_masked = self.prob_mask(span_prob, span_mask)  # (batch_size, span_num, span_label_num)
        if is_train:
            num_spans_to_keep = self.get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))
            top_v = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)[0][:, -1:]
            top_mask = span_prob[:, :, self.ner_neg_id] <= -top_v  # (batch_size, span_num)
            span_mask_subset = span_mask * (top_mask | span_ner_labels.ne(self.ner_neg_id)).float()
        else:
            span_mask_subset = span_mask

        span_len = span_ids[:, :, 1] - span_ids[:, :, 0] + 1
        if is_train:
            span_loss = sequence_cross_entropy_with_logits(span_logits, span_ner_labels, span_mask_subset, average='sum')
        else:
            self.ner_prf(span_logits.max(-1)[1], span_ner_labels, span_mask_subset.long(), bucket_value=span_len)
        # span pair (relation)
        # select span pairs by span scores to construct embedding
        num_spans_to_keep = self.get_num_spans_to_keep(self.spans_per_word, seq_len, span_prob.size(1))
        # SHAPE: (task_batch_size, num_spans_to_keep)
        _, top_ind = (-span_prob_masked[:, :, self.ner_neg_id]).topk(num_spans_to_keep, -1)
        # sort according to the order (not strictly based on order because spans overlap)
        top_ind = top_ind.sort(-1)[0]

        # get out-of-bound mask
        # TODO: span must be located at the beginning
        # SHAPE: (task_batch_size, num_spans_to_keep)
        top_ind_mask = top_ind < span_mask.sum(-1, keepdim=True).long()

        # get pairs
        spans_num = span_ids.size(1)
        external2internal = self.extenral_to_internal(top_ind, spans_num)

        # SHAPE: (batch_size * num_span_pairs * 2)
        span_pairs, span_pair_mask, span_pair_shape = self.span_ind_to_pair_ind(
            top_ind, top_ind_mask, method=None, absolute=False)

        span_pairs_internal = external2internal(span_pairs)

        # get negative span logits of the pair by sum
        # SHAPE: (batch_size * num_span_pairs * 2)
        flat_span_pairs = flatten_and_batch_shift_indices(span_pairs, spans_num)
        # SHAPE: (batch_size, num_span_pairs, 2)
        span_pair_len = batched_index_select(
            span_len.unsqueeze(-1), span_pairs, flat_span_pairs).squeeze(-1)
        # SHAPE: (batch_size, num_span_pairs)
        span_pair_len = span_pair_len.max(-1)[0]

        # get span kept
        # SHAPE: (batch_size * num_spans_to_keep)
        flat_top_ind = flatten_and_batch_shift_indices(top_ind, spans_num)
        # SHAPE: (batch_size, num_spans_to_keep, span_emb_dim)
        span_emb_for_pair = batched_index_select(span_emb, top_ind, flat_top_ind)
        # span pair prediction
        # SHAPE: (batch_size, num_span_pairs, num_classes)
        span_pair_logits = self.span_pair_label_proj(self.span_pair_layer(span_emb_for_pair, span_pairs_internal))

        span_pair_prob = F.softmax(span_pair_logits, dim=-1)
        _, span_pair_pred = span_pair_prob.max(2)
        span_pair_mask_for_loss = span_pair_mask

        # SHAPE: (task_batch_size, num_span_pairs)
        ref_span_pair_labels = relation_labels
        # SHAPE: (task_batch_size, num_spans_to_keep * num_spans_to_keep)
        span_pair_labels = self.label_span_pair(span_pairs, relation_indices, ref_span_pair_labels, relation_mask)

        if is_train:
            span_pair_loss = sequence_cross_entropy_with_logits(
                span_pair_logits, span_pair_labels, span_pair_mask_for_loss,
                average='sum')
            loss = span_loss + span_pair_loss
            return loss
        else: #直接给P. R. F1
            recall = ref_span_pair_labels.ne(self.re_neg_id)
            recall = (recall.float() * relation_mask).long()
            self.re_prf(span_pair_pred, span_pair_labels, span_pair_mask.long(), recall=recall, bucket_value=span_pair_len)
            return self.ner_prf, self.re_prf

    def metric_reset(self):
        self.ner_prf.reset()
        self.re_prf.reset()
        
    def prob_mask(self,
                  prob: torch.FloatTensor,
                  mask: torch.FloatTensor,
                  value: float = 1.0):
        ''' Add value to the positions masked out. prob is larger than mask by one dim. '''
        return prob + ((1.0 - mask) * value).unsqueeze(-1)

    def get_num_spans_to_keep(self,
                              spans_per_word,
                              seq_len,
                              max_value):

        if type(spans_per_word) is float:
            num_spans_to_keep = max(min(int(math.floor(spans_per_word * seq_len)), max_value), 1)
        elif type(spans_per_word) is int:
            num_spans_to_keep = max(min(spans_per_word, max_value), 1)
        else:
            raise ValueError
        return num_spans_to_keep

    def extenral_to_internal(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             total_num_spans: int,
                             ):  # SHAPE: (batch_size, total_num_spans)
        batch_size, num_spans = span_ind.size()
        # SHAPE: (batch_size, total_num_spans)
        converter = span_ind.new_zeros((batch_size, total_num_spans))
        new_ind = torch.arange(num_spans, device=span_ind.device).unsqueeze(0).repeat(batch_size, 1)
        # SHAPE: (batch_size, total_num_spans)
        converter.scatter_(-1, span_ind, new_ind)
        def converter_(ind):
            flat_ind = flatten_and_batch_shift_indices(ind, total_num_spans)
            new_ind = batched_index_select(converter.unsqueeze(-1), ind, flat_ind).squeeze(-1)
            return new_ind  # the same shape as ind
        return converter_

    def span_ind_to_pair_ind(self,
                             span_ind: torch.LongTensor,  # SHAPE: (batch_size, num_spans)
                             span_ind_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_spans)
                             start_span_ind: torch.LongTensor = None,  # SHAPE: (batch_size, num_spans2)
                             start_span_ind_mask: torch.FloatTensor = None,  # SHAPE: (batch_size, num_spans2)
                             method: str = None,
                             absolute: bool = True):
        ''' Create span pair indices and corresponding mask based on selected spans '''
        batch_size, num_spans = span_ind.size()

        if method and method.startswith('left:'):
            left_size = int(method.split(':', 1)[1])

            # get mask
            # span indices should be in the same order as they appear in the sentence
            if absolute:
                # SHAPE: (batch_size, num_spans, num_spans)
                left_mask = (span_ind.unsqueeze(1) < span_ind.unsqueeze(2)) & \
                            (span_ind.unsqueeze(1) >= (span_ind.unsqueeze(2) - left_size))
            else:
                # SHAPE: (num_spans,)
                end_boundary = torch.arange(num_spans, device=span_ind.device)
                start_boundary = end_boundary - left_size
                # SHAPE: (num_spans, num_spans)
                left_mask = (end_boundary.unsqueeze(0) < end_boundary.unsqueeze(-1)) & \
                            (end_boundary.unsqueeze(0) >= start_boundary.unsqueeze(-1))
                left_mask = left_mask.unsqueeze(0).repeat(batch_size, 1, 1)

            # SHAPE: (batch_size, num_spans)
            left_mask_num = left_mask.sum(-1)
            left_mask_num_max = max(left_mask_num.max().item(), 1)  # keep at least 1 span pairs to avoid bugs
            # SHAPE: (batch_size, num_spans)
            left_mask_num_left = left_mask_num_max - left_mask_num
            # SHAPE: (1, 1, left_mask_num_max)
            left_mask_ext = torch.arange(left_mask_num_max, device=span_ind.device).unsqueeze(0).unsqueeze(0)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            left_mask_ext = left_mask_ext < left_mask_num_left.unsqueeze(-1)
            # SHAPE: (batch_size, num_spans, num_spans + left_mask_num_max)
            left_mask = torch.cat([left_mask, left_mask_ext], -1)

            # extend span_ind and span_ind_mask
            # SHAPE: (batch_size, num_spans + left_mask_num_max)
            span_ind_child = torch.cat([span_ind,
                                        span_ind.new_zeros((batch_size, left_mask_num_max))], -1)
            span_ind_child_mask = torch.cat([span_ind_mask,
                                             span_ind_mask.new_zeros((batch_size, left_mask_num_max))], -1)
            # SHAPE: (batch_size, num_spans, left_mask_num_max)
            span_ind_child = span_ind_child.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)
            span_ind_child_mask = span_ind_child_mask.unsqueeze(1).masked_select(left_mask).view(
                batch_size, num_spans, left_mask_num_max)

            # concat with parent ind
            span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                      span_ind_child], -1)
            span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, left_mask_num_max),
                                          span_ind_child_mask], -1) > 0
            # SHAPE: (batch_size, num_spans * left_mask_num_max, 2)
            span_pairs = span_pairs.view(-1, num_spans * left_mask_num_max, 2)
            # SHAPE: (batch_size, num_spans * left_mask_num_max)
            span_pair_mask = span_pair_mask.view(-1, num_spans * left_mask_num_max, 2).all(-1).float()

            # TODO: Because of span_ind_mask, the result might not have left_size spans.
            #   This problem does not exist when the spans are all located at the top of the tensor
            return span_pairs, span_pair_mask, (num_spans, left_mask_num_max)

        if method == 'gold_predicate':
            _, num_spans2 = start_span_ind.size()
            # default: compose num_spans2 * num_spans pairs
            span_pairs = torch.stack([start_span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            span_pair_mask = torch.stack([start_span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                          span_ind_mask.unsqueeze(1).repeat(1, num_spans2, 1)], -1)
            # SHAPE: (batch_size, num_spans2 * num_spans, 2)
            span_pairs = span_pairs.view(-1, num_spans2 * num_spans, 2)
            # SHAPE: (batch_size, num_spans * num_spans)
            span_pair_mask = span_pair_mask.view(-1, num_spans2 * num_spans, 2).all(-1).float()
            return span_pairs, span_pair_mask, (num_spans2, num_spans)

        # default: compose num_spans * num_spans pairs
        span_pairs = torch.stack([span_ind.unsqueeze(2).repeat(1, 1, num_spans),
                                  span_ind.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        span_pair_mask = torch.stack([span_ind_mask.unsqueeze(2).repeat(1, 1, num_spans),
                                      span_ind_mask.unsqueeze(1).repeat(1, num_spans, 1)], -1)
        # SHAPE: (batch_size, num_spans * num_spans, 2)
        span_pairs = span_pairs.view(-1, num_spans * num_spans, 2)
        # SHAPE: (batch_size, num_spans * num_spans)
        span_pair_mask = span_pair_mask.view(-1, num_spans * num_spans, 2).all(-1).float()
        return span_pairs, span_pair_mask, (num_spans, num_spans)

    def label_span_pair(self,
                        span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs1, 2)
                        ref_span_pairs: torch.IntTensor,  # SHAPE: (batch_size, num_span_pairs2, 2)
                        ref_span_pair_labels: torch.IntTensor,  # SHPAE: (batch_size, num_span_pairs2)
                        ref_span_pair_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_span_pairs2)
                        spans: torch.IntTensor = None,  # SHAPE: (batch_size, num_spans, 2)
                        use_binary: bool = False,
                        span_pair_pred: torch.IntTensor = None  # SHAPE: (batch_size, num_span_pairs1)
                        ): # SHPAE: (batch_size, num_span_pairs1)
        neg_label_ind = self.re_neg_id
        device = span_pairs.device
        span_pairs = span_pairs.cpu().numpy()
        ref_span_pairs = ref_span_pairs.cpu().numpy()
        ref_span_pair_labels = ref_span_pair_labels.cpu().numpy()
        ref_span_pair_mask = ref_span_pair_mask.cpu().numpy()
        batch_size = ref_span_pairs.shape[0]
        ref_num_span_pairs = ref_span_pairs.shape[1]
        num_span_pairs = span_pairs.shape[1]

        if spans is not None and use_binary:
            spans = spans.cpu().numpy()
            if span_pair_pred is not None:
                span_pair_pred = span_pair_pred.cpu().numpy()

        span_pair_labels = []
        for b in range(batch_size):
            label_dict = defaultdict(lambda: neg_label_ind)
            label_dict.update(dict((tuple(ref_span_pairs[b, i]), ref_span_pair_labels[b, i])
                                   for i in range(ref_num_span_pairs) if ref_span_pair_mask[b, i] > 0))
            labels = []
            for i in range(num_span_pairs):
                tsp1, tsp2 = tuple(span_pairs[b, i])
                assign_label = label_dict[(tsp1, tsp2)]
                if span_pair_pred is not None:
                    pred_label = span_pair_pred[b, i]
                else:
                    pred_label = None
                if pred_label == neg_label_ind:  # skip pairs not predicated as positive
                    labels.append(assign_label)
                    continue
                if spans is not None and use_binary:
                    # find overlapping span pairs
                    has_overlap = False
                    for (sp1, sp2), l in label_dict.items():
                        if l == neg_label_ind:
                            continue
                        if pred_label and l != pred_label:  # only look at ground truth with predicted label
                            continue
                        if self.has_overlap(spans[b, tsp1], spans[b, sp1]) and \
                                self.has_overlap(spans[b, tsp2], spans[b, sp2]):
                            assign_label = l
                            has_overlap = True
                labels.append(assign_label)
            span_pair_labels.append(labels)
        return torch.LongTensor(span_pair_labels).to(device)
