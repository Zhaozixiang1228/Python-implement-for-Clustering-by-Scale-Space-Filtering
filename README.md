# Python implement for Clustering by Scale-Space Filtering
**Codes for *"Clustering by Scale-Space Filtering" (IEEE TPAMI 2000)***

## Citation

Yee Leung, Jiang-She Zhang and Zong-Ben Xu, "Clustering by scale-space filtering," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no. 12, pp. 1396-1410, Dec. 2000, doi: 10.1109/34.895974.

[*[Paper]*](https://ieeexplore.ieee.org/document/895974)

Cite:

```
@ARTICLE{895974,
    author={ {Yee Leung} and  {Jiang-She Zhang} and  {Zong-Ben Xu}},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Clustering by scale-space filtering}, 
    year={2000},
    volume={22},
    number={12},
    pages={1396-1410},
    doi={10.1109/34.895974}}
```

## Abstract

In pattern recognition and image processing, the major application areas of cluster analysis, human eyes seem to possess a singular aptitude to group objects and find important structures in an efficient and effective way. Thus, a clustering algorithm simulating a visual system may solve some basic problems in these areas of research. From this point of view, we propose a new approach to data clustering by modeling the blurring effect of lateral retinal interconnections based on scale space theory. In this approach, a data set is considered as an image with each light point located at a datum position. As we blur this image, smaller light blobs merge into larger ones until the whole image becomes one light blob at a low enough level of resolution. By identifying each blob with a cluster, the blurring process generates a family of clustering along the hierarchy. The advantages of the proposed approach are: 1) The derived algorithms are computationally stable and insensitive to initialization and they are totally free from solving difficult global optimization problems. 2) It facilitates the construction of new checks on cluster validity and provides the final clustering a significant degree of robustness to noise in data and change in scale. 3) It is more robust in cases where hyperellipsoidal partitions may not be assumed. 4) it is suitable for the task of preserving the structure and integrity of the outliers in the clustering process. 5) The clustering is highly consistent with that perceived by human eyes. 6) The new approach provides a unified framework for scale-related clustering algorithms derived from many different fields such as estimation theory, recurrent signal processing on self-organization feature maps, information theory and statistical mechanics, and radial basis function neural networks.

## Model Implement 

### Algorithm pseudo-code

<img src="sources\SSF1.png" style="zoom:80%;" />

###  Evolutionary tree of cluster centers obtained by algorithm.

<img src="sources\SSF2.jpg" style="zoom:90%;" />

### Gif Exhibition of Scale-Space Filtering in clustering

<img src="sources\Clustering Results by Scale-Space Filtering.gif" alt="Clustering Results by Scale-Space Filtering" style="zoom:80%;" />

### Number of blob center VS. Iteration number

<img src="sources\Number of blob center.png" alt="Number of blob center" style="zoom:50%;" />