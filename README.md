# PCT: Point Cloud Transformer

This is a Jittor implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf


## <font color=red>News</font> :

* 2021.3.31 : We try to add simple position embedding in each self-attention layer, we get a more stable training process and 93.3% (5 run best) accuracy on modelnet40 dataset. Code updates in classification network.
* 2021.3.29 : PCT has been accepted by Computational Visual Media Journal (CVMJ).


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

##### Version 1 : https://github.com/Strawberry-Eat-Mango/PCT_Pytorch (Pytorch version with classification acc 93.2% on ModelNet40)
##### Version 2 : https://github.com/qq456cvb/Point-Transformers (Pytorch version with classification acc 92.6% on ModelNet40)
#### About part segmentation, if you want to reproduce the part segmentation results, you can refer this : https://github.com/AnTao97/dgcnn.pytorch 

<!-- ## Description -->


<!-- Now, we only release the core code of our paper. All code and pretrained models will be available soon.
 -->
## Citation

If it is helpful for your work, please cite this paper:
```
@article{Guo_2021,
   title={PCT: Point cloud transformer},
   volume={7},
   ISSN={2096-0662},
   url={http://dx.doi.org/10.1007/s41095-021-0229-5},
   DOI={10.1007/s41095-021-0229-5},
   number={2},
   journal={Computational Visual Media},
   publisher={Springer Science and Business Media LLC},
   author={Guo, Meng-Hao and Cai, Jun-Xiong and Liu, Zheng-Ning and Mu, Tai-Jiang and Martin, Ralph R. and Hu, Shi-Min},
   year={2021},
   month={Apr},
   pages={187–199}
}
```
