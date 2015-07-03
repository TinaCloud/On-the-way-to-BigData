The algorithms listed in that paper are: 

- trees, 
- k-means, 
- support vector machines, 
- Apriori, 
- Expectation Maximization (EM), 
- PageRank, 
- AdaBoost, 
- k-Nearest Neighbors, 
- Naïve Bayes, 
- CART. 
 
## SVM
大家对SVM的基本原理普遍表述为，SVM通过非线性变换把原空间映射到高维空间，然后在这个高维空间构造线性分类器，因为在高维空间数据点更容易分开。甚至有部分学者认为SVM可以克服维数灾难(curse of dimensionality)。

but, SVM在高维空间里构建分类器后，为什么这个分类器不会对原空间的数据集Overfitting呢？

SVM的成功:

- SVM求解最优分类器的时候，使用了L2-norm regularization，这个是控制Overfitting的关键.
- SVM不需要显式地构建非线性映射，而是通过Kernel trick完成，这样大大提高运算效率。
- SVM的优化问题属于一个二次规划（Quadratic programming），优化专家们为SVM这个特殊的优化问题设计了很多巧妙的解法，比如SMO（Sequential minimal optimization）解法。
- Vapnika的统计学习理论为SVM提供了很好的理论背景.

## Boosted Trees
它基本的想法是通过对弱分类器的组合来构造一个强分类器。AdaBoost把多个不同的决策树用一种非随机的方式组合起来，表现出惊人的性能！第一，把决策树的准确率大大提高，可以与SVM媲美。第二，速度快，且基本不用调参数。第三，几乎不Overfitting。

## Least angle regression (LAR)

LAR把Lasso （L1-norm regularization）和Boosting真正的联系起来. LAR开启了一个光明的时代：有关Sparsity的好文章如雨后春笋般地涌现，比如Candes和Tao的Dantzig Selector。LAR的成功除了它漂亮的几何性质之外，还有它的快速算法。

[模糊系统 and statistical learning](http://cos.name/2011/12/stories-about-statistical-learning/)
## Coordinate Descent / Friedman


## LASSO (the least absolute shrinkage and selection operator) / Tibshrani

Tibshrani自己说他的Lasso是受到Breiman的Non-Negative Garrote（NNG）的启发。 Lasso把NNG的两步合并为一步，即L1-norm regularization。Lasso的巨大优势在于它所构造的模型是Sparse的，因为它会自动地选择很少一部分变量构造模型。

## Non-Negative Garrote (NNG) / Breiman
Breiman: random forest

Friedman
