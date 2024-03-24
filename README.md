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
| SciERC   | 70.98（best）   | 46.19   |
| SciERC   | 69.87   | 47.27（best）   |
| NYT24(NYT)   | 96.52（best）   | 84.90   |
| NYT24(NYT)   | 96.21   | 85.06（best）   |
| NYT29   | 内容6   | 内容6   |
| WebNLG   | 97.94   | 92.64(best)|
| ACE2004_fold1   | 内容5   | 内容6   |
| ACE2004_fold2   | 90.11(best)   | 51.11   |
| ACE2004_fold2   | 88.04   | 53.98(best)   |
| ACE2004_fold3   | 90.00   | 42.39   |
| ACE2004_fold3   | 88.86   | 46.77(best)   |
| ACE2004_fold4   | 92.03(best)   | 46.72   |
| ACE2004_fold4   | 91.04   | 48.67(best)   |
| ACE2004_fold4   | 内容5   | 内容6   |
| ACE2004_fold4   | 内容5   | 内容6   |
| ACE2005   | 90.12(best)  | 62.43   |
| ACE2005   | 89.51  | 64.84(best)   |

