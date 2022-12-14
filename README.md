# Synthetic Tumors Make AI Segment Tumors Better

This repository provides extensive examples of synthetic liver tumors generated by our novel strategies. Check to see if you could tell which is real tumor and which is synthetic tumor. More importantly, our synthetic tumors can be used for training AI models, and have proven to achieve a similar (actually, *better*) performance in real tumor segmentation than a model trained on real tumors. 

**Amazing**, right? 

<p align="center"><img width="100%" src="figures/VisualTuringTest.png" /></p>

Tumor generation code will be release in a few months or please make a request to Dr. Zongwei Zhou ([zzhou82@jh.edu](mailto:zzhou82@jh.edu)).

## Paper

<b>Synthetic Tumors Make AI Segment Tumors Better</b> <br/>
Qixin Hu<sup>1</sup>, [Junfei Xiao](https://lambert-x.github.io/)<sup>2</sup>, [Yixiong Chen](https://scholar.google.com/citations?hl=en&user=bVHYVXQAAAAJ)<sup>3</sup>, Shuwen Sun<sup>4</sup>, [Jie-Neng Chen](https://scholar.google.com/citations?hl=en&user=yLYj88sAAAAJ)<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>2,*</sup> <br/>
<sup>1 </sup>Huazhong University of Science and Technology,   <sup>2 </sup>Johns Hopkins University, <br/>
<sup>3 </sup>Fudan University,   <sup>4 </sup>The First Affiliated Hospital of Nanjing Medical University <br/>
Medical Imaging Meets NeurIPS, 2022 <br/>
[paper](https://arxiv.org/pdf/2210.14845.pdf) | [code](https://github.com/MrGiovanni/SyntheticTumors) | [slides]() | [demo]()

## TODO

- [x] Upload the paper to arxiv
- [ ] Make a video about Visual Turing Test (will appear in YouTube)
- [ ] Make an online app for Visual Turing Test
- [x] Apply for a US patent

## Citation

```
@article{hu2022synthetic,
  title={Synthetic Tumors Make AI Segment Tumors Better},
  author={Hu, Qixin and Xiao, Junfei and Chen, Yixiong and Sun, Shuwen and Chen, Jie-Neng and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2210.14845},
  year={2022}
}
```

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research. The segmentation model is based on [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb); we appreciate the effort of the authors for providing open source code to the community.
