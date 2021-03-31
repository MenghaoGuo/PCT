# PCT: Point Cloud Transformer

This is a Jittor implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf


## News:

* <font color="#dd0000"> 2021.3.31 </font><br /> 
: We try to add simple position embedding in each self-attention layer, we get a more stable training process and 93.3% accuracy on modelnet40 dataset. code updates in classification network.
* 2021.3.29 : PCT has been accepted by CVMJ.


## Astract


The irregular domain and lack of ordering make it challenging to design deep neural networks for point cloud processing. This paper presents a novel framework named Point Cloud Transformer(PCT) for point cloud learning. PCT is based on Transformer, which achieves huge success in natural language processing and displays great potential in image processing. It is inherently permutation invariant for processing a sequence of points, making it well-suited for point cloud learning. To better capture local context within the point cloud, we enhance input embedding with the support of farthest point sampling and nearest neighbor search. Extensive experiments demonstrate that the PCT achieves the state-of-the-art performance on shape classification, part segmentation and normal estimation tasks


![image](https://github.com/MenghaoGuo/PCT/blob/main/imgs/attention.png)


## Architecture


![image](https://github.com/MenghaoGuo/PCT/blob/main/imgs/architecture.png)



## Jittor

Jittor is a  high-performance deep learning framework which is easy to learn and use. It provides interfaces like Pytorch.

You can learn how to use Jittor in following links:

Jittor homepage:  https://cg.cs.tsinghua.edu.cn/jittor/

Jittor github:  https://github.com/Jittor/jittor

If you has any questions about Jittor, you can ask in Jittor developer QQ Group: 761222083

## Other implementation

##### Version 1 : https://github.com/uyzhang/PCT_Pytorch (Pytorch version with classification acc 93.2% on ModelNet40)
##### Version 2 : https://github.com/qq456cvb/Point-Transformers (Pytorch version with classification acc 92.6% on ModelNet40)


## Description


Now, we only release the core code of our paper. All code and pretrained models will be available soon.

## Citation

If it is helpful for your work, please cite this paper:
```
@misc{guo2021pct,
      title={Pct: Point cloud transformer}, 
      author={Meng-Hao Guo and Jun-Xiong Cai and Zheng-Ning Liu and Tai-Jiang Mu and Ralph R. Martin and Shi-Min Hu},
      year={2021},
      eprint={2012.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
