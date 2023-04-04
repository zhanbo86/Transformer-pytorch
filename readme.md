# Attention is all you need & Transformer : A Pytorch Implementation for Education


## Introduction
Realize the tranformer network following the paper "attention is all you need" strictly except two differencies:
1. Moving all layernorms from after sublayers to before sublayers, this accelerate training speed significantly.
2. Addding a dropout after the last linear layer in Feed-Forward sublayer refer to the transformer implementation in pytorch (see torch.nn.Transformer). The benefit is not obvious in my test.

---------------------------------------------------
paper link: ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)

The Transformer-model architecture:
![The Transformer-model architecture](./transformer.png)

# Requirements
`pip install -r requirements.txt`

If torchdata is conflicted with other libs, you can build torchdata from source to solve it.

# Usage
1. Use completed transformer, including both encoder and decoder modules, for tasks such as translation.
- Download dataset (USE WMT'16 Multimodal Translation:de-en)

  `python -m spacy download de_core_news_sm`
  `python -m spacy download en_core_web_sm`

- Train model

  `python train.py --net transformer`

2. Use transfomer only includng decoder module for tasks such as GPT.
- Download dataset (USE tiny shakespear dataset)

  `wegt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

- Train model

  `python train.py --net decoder`


# Acknowledgement
1. The project structure, some scripts are partly borrowed from (https://github.com/jadore801120/attention-is-all-you-need-pytorch) and Andrej Karpathy youtube video "Let's build GPT: from scratch, in code, spelled out"(https://www.youtube.com/watch?v=kCc8FmEb1nY).
2. The WMT'16 dataset preprocessing steps are partly borrowed from (https://github.com/jadore801120/attention-is-all-you-need-pytorch).
3. The tiny shakespear dataset preprocessing steps are heavily borrowed from  "Let's build GPT: from scratch, in code, spelled out"(https://www.youtube.com/watch?v=kCc8FmEb1nY).







