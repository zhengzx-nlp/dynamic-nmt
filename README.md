# dynamic-nmt
Pytorch implementation of EMNLP paper "Dynamic past and future for neural machine translation"


This repo is based on [NJUNMT-pytorch](https://github.com/whr94621/NJUNMT-pytorch).

## Requirement
`pytorch=0.4.1`


## Training 
```bash
bash train_capsule_transformer_wp_bca.sh
```

## Decoding
```bash
bash translate.sh
```


## Citation
```
@inproceedings{zheng2019dynamic,
  title={Dynamic Past and Future for Neural Machine Translation},
  author={Zheng, Zaixiang and Huang, Shujian and Tu, Zhaopeng and DAI, XIN-YU and Jiajun, CHEN},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={930--940},
  year={2019}
}
```
