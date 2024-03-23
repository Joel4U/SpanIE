# SpanIE
A span-based joint named entity recognition (NER) and relation extraction model with RoBERTa

This code repository has been restructured based on [JointIE](https://github.com/JiachengLi1995/JointIE/) for the purpose of dependency transition models. It mainly references the following models and codes:

Generalizing Natural Language Analysis through Span-relation Representations (ACL2020). [paper] (https://arxiv.org/abs/1911.03822)

LSTM/BERT-CRF Model for Named Entity Recognition (or Sequence Labeling) [code](https://github.com/allanj/pytorch_neural_crf)

# Environment
torch==2.2.0

Transformer==4.37.2

python==3.8.0

# Train and Evaluate Models

Train and evaluate model with default configure.（RoBERTa-Large, Learing rate 1e-5）


```bash
python transformers_trainer.py --dataset scierc
```

# Results with Default Configure on Test Set
| Dataset | NER (F1)	 | Relation (F1) |
|---------|---------|---------|
| Wet Lab Protocols   | 内容2   | 内容3   |
| SciERC   | 75.76（best）   | 47.02   |
| SciERC   | 74.25   | 49.90（best）   |
| NYT24(NYT)   | 内容5   | 内容6   |
| NYT29   | 内容5   | 内容6   |
| WebNLG   | 97.94   | 92.64(best)|
| ACE2004   | 内容5   | 内容6   |
| ACE2005   | 90.12(best)  | 62.43   |
| ACE2005   | 89.51  | 64.84(best)   |

