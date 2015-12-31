# Data mining algorithm
- supervised learning
	- classification
		- model based
			- logistic regression
			- LDA/QDA
		- model-free approaches
			- Tree
			- KNN
			- Bayes rule
			- SVM
			- Neural networks
			- Random Forest
		- Examples
			- fraud detection
			- spam/nonspam email
			- cancer detection
	- regression
		- Regression
		- CART
		- Neural networks
		- SVM
		- Examples	
			- wine quality
			- search quality vs stock price
			- recommendation: 一般来说，电商的“猜你喜欢”（即推荐引擎）都是在协同过滤算法（Collaborative Filter）的基础上，搭建一套符合自身特点的规则库。即该算法会同时考虑其他顾客的选择和行为，在此基础上搭建产品相似性矩阵和用户相似性矩阵。基于此，找出最相似的顾客或最关联的产品，从而完成产品的推荐。
- unsupervised learning
	- association analysis
		- Examples
			- walmart example beer and diaper
			- social networks
	- clustering
		- k-means
		- hierarchical clustering
		- MSD, density
		- Examples
			- text mining
			- 识别不同的客户群体，然后针对不同的客户群体，精准地进行产品设计和推送，

## Examples
1. Social network
种子客户和社会网络，最早出现在电信领域的研究。即，通过人们的通话记录，就可以勾勒出人们的关系网络。电信领域的网络，一般会分析客户的影响力和客户流失、产品扩散的关系。

基于通话记录，可以构建客户影响力指标体系。采用的指标，大概包括如下，一度人脉、二度人脉、三度人脉、平均通话频次、平均通话量等。基于社会影响力，分析的结果表明，高影响力客户的流失会导致关联客户的流失。其次，在产品的扩散上，选择高影响力客户作为传播的起点，很容易推动新套餐的扩散和渗透。

此外，社会网络在银行（担保网络）、保险（团伙欺诈）、互联网（社交互动）中也都有很多的应用和案例。

2. 字符识别
手机拍照时会自动识别人脸，还有一些APP，例如扫描王，可以扫描书本，然后把扫描的内容自动转化为word。这些属于图像识别和字符识别（Optical Character Recognition）。图像识别比较复杂，字符识别理解起来比较容易些。

查找了一些资料，字符识别的大概原理如下，以字符S为例。第一，把字符图像缩小到标准像素尺寸，例如12*16。注意，图像是由像素构成，字符图像主要包括黑、白两种像素。

第二，提取字符的特征向量。如何提取字符的特征，采用二维直方图投影。就是把字符（12*16的像素图）往水平方向和垂直方向上投影。水平方向有12个维度，垂直方向有16个维度。这样分别计算水平方向上各个像素行中黑色像素的累计数量、垂直方向各个像素列上的黑色像素的累计数量。从而得到水平方向12个维度的特征向量取值，垂直方向上16个维度的特征向量取值。这样就构成了包含28个维度的字符特征向量。

第三，基于前面的字符特征向量，通过神经网络学习，从而识别字符和有效分类。

3. 文学著作与统计：红楼梦归属
总而言之，主要通过一些指标量化，然后比较指标之间是否存在显著差异，藉此进行写作风格的判断。

## Multivariate method analysis
- dimensionality reduction
- classification
- clustering

multivariate methods rely on distances between data points. Distance can be measured in different ways. 

standardized distance: we wish to give more weight in the distance calculation to variables that are measured more precisely. And the weights are inversely proportional to the standard deviation in the measurements. example is in the Gaussian density distribution.

when the measurements are correlated, we can construct a statistical distance that accounts for correlations and unequal variances by:
- first rotating the axes to be parallel to the axes of the ellipsoid. 
- then using the expression for a standardized distance
- Mahalnobis distance
- thus, the quadratic form can be used to express distance

Visualization of the multivariate analysis
- pairwise scatter plot
- 3-d plot
- survey plot

Multivariate Normal Distribution
we will almost always assume that the joint distribution of the measures on each sample is the p-dimensiional multivariate normal distribution because of the central limit theorem.

if (lambda,p) is an eigenvalue-eigenvector pair for covariance matrix, then (lambda^{-1},p) is an eigenvalue-eigenvector pair of inverse covariance matrix.

if sigma_12 >0, the major axis of the ellipse will be in the direction of the 45 degree line. The actual value of sigma_12 does not matter. If sigma_12 < 0, then the major axis will be perpendicular to the 45 degree line.

Them: x~MVN iff a^{T}X is normally distributed.

Assessing multivariate normality
- pairwise plot
- marginal distribution
- conditional distribution
- outliers?
- Q-Q plot
- Shapiro-Wilks' Test
- Anderson-Darling Test
- Kolmogorov-Smirnov Statistic
- Chi Square plot
	- d^2 vs. quantiles

Transformation
- right skewed data: log
- counts: square root
- proportions: logit
- correlation: Fisher's z(r)=0.5*log[(1+r)/(1-r)]
- Boc-Cox


However, these checks do not assure the normality distribution.

multivariate tests
the Hotelling T^2 test is the most powerful test in the class of tests that are invariate to full rank linear transformations.

The combined set of individual intervals result in a simultaneous confidence level that is less than the nominal 1-alpha.
- Hoteling T^2
- Bonferroni

In the special case where the covariance matrix is diagonal, the joint coverage prob of p ordinary t intervals is (1-alpha)^p. Clearly, to guarantee 1- alpha joint coverage probability, the t intervals need to be made wider.

Paired comparison design
- we measure volumes of sales of a certain product in a certain market before and after an advertising campaign.
- we count traffic accidents at a collection of intersections when stop sign controls or light controls were used for signalization.

In the paired comparison design, we calculate the difference D. The average D is normally distributed. We are testing if mu_D=0 or not.

But in the LISA project, we are interested in the agreement between SP and faculty. The agreement is either 0 or 1.  

Repeated measure design




## Q： Loss function
The smaller the loss function, the better the model
http://52opencourse.com/125/coursera%E5%85%AC%E5%BC%80%E8%AF%BE%E7%AC%94%E8%AE%B0-%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%AC%E5%85%AD%E8%AF%BE-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92-logistic-regression

- 0-1 loss function
	- L(f(x),y)=1 when y!= f(x) and 0 when y = f(x)
- MSE (quadratic loss function)
- absolute 
- log-likelihood 
	- L(f(x),y) = -log P(Y|X)
- hinge 
	- SVM
- exp-loss
	- Boosting

the expected loss: risk

The problem is to know the conditional probability P(Y|X) (for discriminative model), equivalent to joint distribution P(X,Y) (for regression).

The expected risk vs. the empirical risk

minimum empirical risk
- MLE when loss is log-loss

## Q: Statistical learning Elements
- model
	- f(Y|X)
- strategy
	- regularized loss function
- algorithm
	- optimization method
- Diagnoistic
	- training error
	- test error
		- when it is 0-1 loss function, error rate
	- generalization error
		- because the test set is limited
		- generalization error bound = training error + $$\sqrt {\frac{1}{2N} (log d + log \frac{1}{\delta})} $$ 
		
How to evaluate the classification?
- precision, true positive rate
- accuracy

## Q: Regularization func
- parameter norm
	- zero
		- L0范数是指向量中非0的元素的个数, 就是希望W的大部分元素都是0.
	- one: absolute
	- two: quadratic
	- trace
	- Frobenius
	- nuclear norm: 矩阵奇异值的和,约束Low-Rank.

任何的规则化算子，如果他在Wi=0的地方不可微，并且可以分解为一个“求和”的形式，那么这个规则化算子就可以实现稀疏。

Q: 既然L0可以实现稀疏，为什么不用L0，而要用L1呢？
A: 个人理解一是因为L0范数很难优化求解（NP难问题），二是L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解。

L1: feature selection, and interpretation
L2: overfitting, and condition number, 让我们的优化求解变得稳定和快速

How to reduce the effect of overfitting
1. 在gradient descent中，到目前为止，我们只是解释了L2正则化项有让w“变小”的效果，但是还没解释为什么w“变小”可以防止overfitting？一个所谓“显而易见”的解释就是：更小的权值w，从某种意义上说，表示网络的复杂度更低，对数据的拟合刚刚好（这个法则也叫做奥卡姆剃刀），

2. 过拟合的时候，拟合函数的系数往往非常大，为什么？如下图所示，过拟合，就是拟合函数需要顾忌每一个点，最终形成的拟合函数波动很大。在某些很小的区间里，函数值的变化很剧烈。这就意味着函数在某些小区间里的导数值（绝对值）非常大，由于自变量值可大可小，所以只有系数足够大，才能保证导数值很大。

3. L1 也可以认为是 reduce overfitting，as a special case of beta=0 in the ridge regression

Problems in optimization
1. local minimum
2. ill condition
	1. multicollinearity
	2. condition number值小的就是well-conditioned的，大的就是ill-conditioned的。If matrix A is singular, then the conditional number is infinite.
	3. To caluclate the condition number, we need norm and machine epsion. 我们知道矩阵是没有大小的，范数就相当于衡量一个矩阵的大小.

![](http://i.imgur.com/ksZiNTx.png)

![](http://i.imgur.com/zrpw94u.png)
conditionnumber是一个矩阵（或者它所描述的线性系统）的稳定性或者敏感度的度量，如果一个矩阵的condition number在1附近，那么它就是well-conditioned的，如果远大于1，那么它就是ill-conditioned的，如果一个系统是ill-conditioned的，它的输出结果就不要太相信了。

convex and strong convex
![](http://i.imgur.com/IEFVj52.png)

condition number 太大仍然会导致问题：它会拖慢迭代的收敛速度，如果要获得strongly convex怎么做？最简单的就是往里面加入一项(α/2)*||w||2。

呃，讲个strongly convex花了那么多的篇幅。实际上，在梯度下降中，目标函数收敛速率的上界实际上是和矩阵XTX的 condition number有关，XTX的 condition number 越小，上界就越小，也就是收敛速度会越快。

L1就是按绝对值函数的“坡”下降的，而L2是按二次函数的“坡”下降。所以实际上在0附近，L1的下降速度比L2的下降速度要快。所以会非常快得降到0。

![](http://i.imgur.com/dFhSc8O.png)

L1会趋向于产生少量的特征，而其他的特征都是0，而L2会选择更多的特征，这些特征都会接近于0。

other methods:
in neural networks, dropout some units in a middel layer.

Nuclear form:
rank() 和||W||* 的关系和L0与L1的关系一样。因为rank()是非凸的，在优化问题里面很难求解，那么就需要寻找它的凸近似来近似它了。对，你没猜错，rank(w)的凸近似就是核范数||W||*。

Examples:
1. Matrix Completion
http://blog.csdn.net/zouxy09/article/details/24972869

但如果我们已知A的秩rank(A)<<m且rank(A)<<n，那么我们可以通过矩阵各行(列)之间的线性相关将丢失的元素求出。你会问，这种假定我们要恢复的矩阵是低秩的，合理吗？实际上是十分合理的，比如一个用户对某电影评分是其他用户对这部电影评分的线性组合。所以，通过低秩重构就可以预测用户对其未评价过的视频的喜好程度。从而对矩阵进行填充。

2. Robust PCA
assume the error is sparse. 

由于rank和L0范数在优化上存在非凸和非光滑特性，所以我们一般将它转换成求解以下一个松弛的凸优化问题：

美图
背景建模

3. hyper-parameter tuning

两种：一是尽量测试7个比较靠谱的λ，或者说λ的搜索空间我们尽量广点，所以一般对λ的搜索空间的选择一般就是2的多少次方了，从-10到10啊什么的。但这种方法还是不大靠谱，最好的方法还是尽量让我们的模型训练的时间减少。例如假设我们优化了我们的模型训练，使得我们的训练时间减少到2个小时。那么一个星期我们就可以对模型训练7*24/2=84次，也就是说，我们可以在84个λ里面寻找最好的λ。这让你遇见最好的λ的概率就大多了吧。这就是为什么我们要选择优化也就是收敛速度快的算法，为什么要用GPU、多核、集群等来进行模型训练、为什么具有强大计算机资源的工业界能做很多学术界也做不了的事情（当然了，大数据也是一个原因）的原因了。


## Q: Optimization methods
- Gradient descent
- conjugate gradient method
- Quasi-Newton method
- BFGS method
- L-BFGS (limited-memory BFGS)

算法收敛性的证明以及收敛速度证明

http://dataunion.org/20792.html
2013年3月，谷歌以重金收购DNNresearch的方式请到了Geoffrey Hinton教授（上文提到的深度学习技术的发明者）；2013年12月，Facebook成立了人工智能实验室，聘请了卷积神经网络最负盛名的研究者、纽约大学终身教授Yann LeCun为负责人；2014年5月，有“谷歌大脑之父”美称的Andrew NG（吴恩达）加盟百度，担任首席科学家，负责百度研究院的领导工作，尤其是“百度大脑”计划。这几位人工智能领域泰斗级人物的加入，充分展示了这些互联网巨头对人工智能领域志在必得的决心。


structured data
unstructured data

Occam's razor: Among competing hypotheses, the one with the fewest assumptions should be selected.

奥卡姆剃刀原理: 模型的复杂度必须与问题匹配的。有多种模型能解释数据的时候，就选择最简单的一个. 当两个假说具有完全相同的解释力和预测力时，我们以那个较为简单的假说作为讨论依据.


## Applications:
Classification
1. In bank, classify the customers to decide the loan risk
2. In Internet security, classify the log data to detect the hack
3. In image processing, classify the people face
4. In text recognition, classify the digits
5. In Internet search, classify the web pages, index and ranking
6. In text classification, classify the text based on features including the content, positions of opinions. 
7. tagging: hidden Markov chain, Conditional random fields.

## Perceptron
The hyperplane: wx+b=0, where w is the normal vector, b is the intercept.

n(r-r0)=0, where n = (a,b,c)
so, a(x-x0)+b(y-y0)+c(z-z0)=0
then, the equation of the plane is:
ax + by + cz + (-ax0 - by0 -cz0) = 0

distance between one point and the hyperplane: $|wx+b|/|w|$

Loss function (>0): sum of wrong predictions (not here), but sum of the distance between points and the hyperplane, which is
$$
\sum y(wx+b)/|w|
$$

Algorithm
1. gradient descent
for perceptron, 采用不同的初值 和 不同的无分类点，解可以不一样. That is, the solution to the perceptron is dependent on the initial value and the order of the mistaken data point. 

Intuition: 当一个实例点被误分时，即位于hyperplane错误的一侧是，调整coeff，使hyper plane向该误分点的一侧移动，减少距离，直至越过hyper plane而正确。

In order to obtain the only hyperplane, SVM,

Algorithm evaluation

误分类次数k has a upper bound, k <= (R/r)^2, where R = max ||x_{i}||, and r is the minimum distance between data point and the hyperplane.
原始形式
对偶形式
Q： what is the difference between 原始形式 and 对偶形式?
not much, right?

## KNN
1. model
- Parameter k
	- if k is small (k=1), overfitting, if k is large (k=N), bias.
	- cross-validation
- distance
	- L2
	- L,infinity: that is, the max distance between coordinates.
	- Minkowski
![](http://i.imgur.com/nzjohs2.png)
Note: distance is dependent on the Lp. Different Lp, different D, then different vote.  

2. loss function:
-  majority vote, equivalent to the 0-1 loss function

3. Algorithm
- linear scan: calculate the distance between the data point and the rest
- kd tree (when n > p)

4. Relation between regression and KNN.
- the regression minimizes the expected error conditioned on x. Relaxing the conditioning at a point to include a region close to a point, and with a 0-1 loss function leads to the nearest-neighbor approach.

5. The curse of dimensionality in kNN classification
- as dimensionality increases, the data-points in the training set become closer to the boundary of the sample space than to any other observation. Consequently, prediction is much more difficult at the edges of the training sample, since some extrapolation in prediction may be nedded. Futhermore, for a p-dimensional input problem, the sampling density is proportional to n^(1/p). Hence the sample size required to maintain the same sampling density as in lower dimensions grows exponentially with the number of dimensions. The kNN approach is therefore not immune to the phenomenon of degraded performance in higher dimensions.
 
## Naive Bayes
1. model
- P(Y=c|X=x) = P(X=x|Y=c)*Prior/Denominator, that is, y = argmax {P(Y=c|X=x)}, where prior from the MLE + Laplace smoothing, 
- Independence assumption in P(X=x|Y=c)

2. loss function
0-1 loss function

3. Optimization

## Decision Tree (CART)
each cell is similar to the conditional probability P(Y|X)
1. model, feature selection
- Information gain
	- entropy, H(p(X=x))= -sum (p logp)
	- conditional entropy, H(p(Y|X)) = sum p(Y) H(p(X|Y=y))
	- information gain (mutual information), the decrease in the uncertainty of Y given a certain feature X, defined as H(Y) - H(Y|X)
- Information gain ratio

2. loss function
entropy

3. Alogrithm: Generating the tree
- ID3
	- information gain
	- over fitting problem

- C4.5
	- information gain ratio

- CART
-  divide the feature space into 2 by choosing optimal splitting variable and splitting point
- for each cell, the average \hat y
1. MSE for regression and 
		
2. Gini index for classification
	- Gini (p) = sum p(1-p)

3. Pruning
loss func + alpha*|number of leaf nodes|

Notes:
- a tolerance threshold for entropy or gini index has to be set up as the stop criteria in the tree generation 
- what happen if two features has the same entroy or gini index?

## Logistic regression and maximum entropy model

maximum entropy model, which says to choose the model with max entropy under constraints.模型要满足constraints，而不确定的部分都是等可能的。最大化原理通过熵的最大化来表示可能性，熵是一个可优化的数值。

H[P(Y|X)]=-sum {P(x) P(y|x) log P(y|x)}

improved iterative scaling

Q: newton's method vs. quasi newton's method

gradient descent (first order derivative)

In newton's method, it computes the inverse matrix of the hessian matrix of the object function, (anology is zero-finding in Newton's method), while in quasi newton's method, it computes the approximate inverse matrix by a pd matrix, then update the pd matrix in each iterate. How to update the pd matrix?
1. DFP algorithm
where the G_k+1 = G_k + P_k + Q_k
2. BFGS algorithm
where B is used to approach H instead of using G to approach H^{-1}
3. Broyden algorithm
It is a linear combination of DFP and BFGS

Newton: x^{k+1} = x^{k} - H^{-1}g, where g is the first derivative function

Q: the relationship between logistic regression and the maximum entropy model

## SVM




## Principal component
PCs are derived from the eigenvectors of covariance matrix.

Y1 is the linear combination of Xi that maximizes variance, and subject to aTa = 1.

Y2 is the linear combination of Xi that maximizes variance, and subject to aTa = 1, and cov(Y1,Y2) = 0.

where a turns out to be the eigenvector of variance Sigma.

where the sum of eigen values are the total variance of X.

Note:
1. to standardize the data, use corr(X) instead of corr(Y)
2. if X are independent, then PCA is not working.
3. the eigenvalue-eigenvector paris of S from basis of PCA
	- the eigenvectors determine the directions of maximum variability
	- the eigenvalues are the variances of the linear combinations.

The assumptions of PCA: + Linearity: the new basis is a linear combination of the original basis + Mean and variance are sufficient statistics: PCA assumes that these statistics totally describe the distribution of the data along the axis (ie, the normal distribution). + Large variances have important dynamics: high variance means signal, low variance means noise. This means that PCA implies that the dynamics has high SNR (signal to noise ratio). + The components are orthonormal

http://www.di.fc.ul.pt/~jpn/r/pca/pca.html#a-not-entirely-successful-example-of-image-processing-and-reduction

Q: How many PCs to keep?
- scree plot: explained variance proportion/cumulative vs PC number
- original data size
- interpretation

PCA does not include Y

Q: what is factor pattern in the factor procedure?

For classification, the objective is to find a function that minimizes the classification error.

### Q：Complex data
- categorical and continuous
- image 


### Q: How to import big data?
1. 可以先用Python processing by line(s)
挑你需要的或summarized/rolled-up的数据写出来
再用pandas 或R做model

2. 如果真需要learn from entire data set 干脆直接
上mahout 或者 spark

首先，为什么会Memory Error？
很显然，你的文件有10G，你的内存有10G吗

那怎样用Python读取？很简单，你肯定知道Python的函数有return功能，你进一步了解yield功能就可以了，他可以让你一条一条数据的读取，读取完一个扔掉上一个

如何在这种情况使用Logistic Regression？
Logistic Regression是最典型的在线算法，他在任何一个软件里面都是数据一个一个放进去算的，你现在需要编写一个LR



Q: Classification in high dimension feature space

n=10,100,1000,10000

p=10,100,1000,

**Steps**：
1. separate data set into 3 parts: train, test, validate
2. check the dim of the data
3. data type: continuous and categorical
4. standardize the data
5. methods include
	1. Lasso multinominal
	2. robust regularized PCA
	3. XGBoost
	4. Variable Selection for High-Dimensional Supervised Classification
		1. 2 classes: FAIR (features annealed independence rule) for feature selection and classification in high dim setting.
		2. 

# clustering
- Goals:
Hundreds of clustering methods which may produce quite different groups; and it may not be possible to cluster the individuals into meaningful groups.

The measured distance could be Euclidean distance, squared Euclidean distance, generalized distance, mean distance, etc.

- Models
- Hierarchical
	- when combining subgroups into groups, which criteria to use: 
		- Single linkage
			- chaining effect, the single linkage has the ability to find non-spherical or non-elliptical clusters.
		- Average linkage
			- produce clusters with similar variances
		- Complete linkage
			- complete linkage produces spherical clusters
		- Wards method
			- tend to produce clusters of same size
		- Centroid method
			- robust to outliers
		- density linkage
		- two-stage density linkage
- Partition
In the k-means, Euclidean distance assumes unit covariance and linear separability?

Q: clustering for categorical data, netflow project in Data Analysis I.
In the data set, there are five important categorical variables SRCIP (Source IP), DSTIP (Destination IP), PROTO (Protocol), SCRPORT (Source Port), DSTPORT (Destination Port). We also noticed that some continuous variables may play important roles. For example, the variable flow time, which is the difference between the flow start and the flow finish. 
Therefore, the problem we are trying to solve is a clustering of mixed data of large size.

In the data set, there are small integer, integer, and big integer for the categorical variables. We tried the function discrete.recode in the package fpc to recode those categorical variables to have standard coding 1,2,3,... Then, We tried the function lcmixed in the package fpc in R. This function provides the necessary information to run an EM-algorithm for MLE for a latent class mixture clustering model, where continuous variables are modeled within the mixture components by Gaussian distributions and categorical variables are modeled within components by independent multinomial distributions. However, R crashed during the run.

We also tried the function kkmeans in the package kernlab in R. The kernlab provides tools for a range of popular machine learning methods including clustering. The kernel method is a partitional clustering method, which can be used to separate data set in sufficiently high dimensional space. The kernel kmeans is able to find complex clusters. It is reported that the kernal kmeans clustering could be used for the data set with mixture variable.

In addition to that, we tried the function kmodes in the package klaR in R. The kmodes methods is said to  be a fast clustering algorithm to cluster categorical variables. However, R crashed again during the run time.

## Comments on clustering
clustering is based on nonmodels, so largely exploratory, so formal inference is not possible. But model-based clustering is an alternative.

The model-based clustering: the sample observations are from a distribution that is a mixture of p components, each component is described by a density function and has an associated probability in the mixture. Thus, the probability model for clustering will often be a mixture of multivariate normal distributions. Each component in the mixture is what we call a cluster.




## Q: Clustering on mixed data? 
- daisy
- agnes
- etc.

## Q: How to tell if data is clustered enough for clustering algorithm to produce meaningful results?

http://stats.stackexchange.com/questions/11691/how-to-tell-if-data-is-clustered-enough-for-clustering-algorithms-to-produce-m/35760#35760

About k-means specifically, you can use the Gap statistics. Basically, the idea is to compute a goodness of clustering measure based on average dispersion compared to a reference distribution for an increasing number of clusters. More information can be found in the original paper:

    Tibshirani, R., Walther, G., and Hastie, T. (2001). Estimating the numbers of clusters in a data set via the gap statistic. J. R. Statist. Soc. B, 63(2): 411-423.

The answer that I provided to a related question highlights other general validity indices that might be used to check whether a given dataset exhibits some kind of a structure.

When you don't have any idea of what you would expect to find if there was noise only, a good approach is to use resampling and study clusters stability. In other words, resample your data (via bootstrap or by adding small noise to it) and compute the "closeness" of the resulting partitions, as measured by Jaccard similarities. In short, it allows to estimate the frequency with which similar clusters were recovered in the data. This method is readily available in the fpc R package as clusterboot(). It takes as input either raw data or a distance matrix, and allows to apply a wide range of clustering methods (hierarchical, k-means, fuzzy methods). The method is discussed in the linked references:

## Q: How to compare performance of two cluster ing methods

http://stats.stackexchange.com/questions/7175/understanding-comparisons-of-clustering-results


## Q: PCA before LDA?


## Q: Inverse of covariance matrix in high dimension space?
Inversion is really sensitive operation that can only be done if the estimate of covariance matrix is really good, which requires more than N data points. Otherwise the covariance matrix will be almost singular.

When the covariance matrix is singular, how to handle this problem?

- Graphical LASSO, (regularization)
- use 
-  before to regularize the problem
	- the following question is that when does PCA help the classification and when hurts?
		- PCA helps: the direction of maximal variance is horizaontal, and the classes are separeted horizontally
		- PCA hurts: the direction of maximal variance is horizontal, but the classes are separeted vertically.


## Methods to reduce the dimensionality
- missing value ratio
- low variance filter
- high correlation filter
- random forests
- PCA
- MDS
- Coorespondence analysis
- factor analysis
- clustering
- Bayesian models
- backward feature elimination
- forward feature construction
- 

## Q: Decomposition
both eigenvectors and eigenvalues are providing us with information about the distortion of a linear transformation: The eigenvectors are basically the direction of this distortion, and the eigenvalues are the scaling factor for the eigenvectors that describing the magnitude of the distortion.

If we are performing the LDA for dimensionality reduction, the eigenvectors are important since they will form the new axes of our new feature subspace; the associated eigenvalues are of particular interest since they will tell us how "informative" the new "axes" are.

