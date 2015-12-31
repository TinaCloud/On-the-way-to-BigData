# **Data Mining**
- Introduction to data mining / Tan, Pang Ning
http://www-users.cs.umn.edu/~kumar/dmbook/index.php

# Chapter 1 Introduction
Data mining is a technology that blends traditional data analysis methods with sophisticated algorithms for processing large volumes of data.

Data mining techniques can be used to support a wide range of business intelligence applications such as customer profiling, targeted marketing, workflow management, store layout, and fraud detection.

## Process of knowledge discovery in databases:

1. Data Preprocessing:
	- Feature selection
	- Dimensionality Reduction
	- Normalization
	- Data Subsetting

2. Data Mining:

3. Postprocessing:
	- Filtering Patterns
	- Visualization
	- Pattern Interpretation

The traditional statistical approach:
- propose a hypothesis,
- design an experiment
- collect and analyze the data,

## Data mining tasks
- descriptive task
	- correlation, e.g., store layout
	- clusters
	- anomalies, e.g., fraud detection
- predictive task
	- classification
	- regression

Example 
[Iris flower data @ UCI ML Repo](https://archive.ics.uci.edu/ml/)

```{r}
library(ggplot2)
data1=read.table("E:/Data/iris.txt",header=F,sep = ",",fill = TRUE)
names(data1)=c('SepalLength','SepalWidth','PetalLength','PetalWidth','class')
summary(data1)
ggplot(data1,aes(PetalLength,PetalWidth,colour=class))+
        # geom_line() +
        geom_point()
```

# Chapter 2 Data
## Type of data

General characteristics of data sets:
- **dimensionality**: # of features
- **sparsity**: in practical terms, sparsity is an advantage because only the non-zero values need to be stored and manipulated.
- **resolution**: 

Data matrix
- the sparse data matrix

Graph-Based Data
- data with relationships among objects
- data with objects that are graphs

Ordered Data
- sequential data (temporal data): with time associated
	- time series data: measurements taken over time
		- notes: temporal autocorrelation: if two measurements are close in time, then the values of those measurements are often very similar.
	- spatial data: 
		- notes: spatial autocorrelation: if two measurements are physically close, then they are similar in other ways as well.
- sequence data: there are positions in an ordered sequence

## Quality of data
**Q: Is the data suitable for its intended use?**
 1. **data cleaning**: the detection and correction of data quality problems
 2. the use of algs that can tolerate poor data quality 

- Noise: 
	- signal processing, robust algorithms that produce acceptable results even when noise is present
- Variance and bias
- Outlier: 
	- unlike noise, outlier can be of interest
- Missing value: 
	- eliminate data objects or features
	- ignore the missing value 
		- cluster
		- classification
	- estimate missing value
		- time series data: interpolation
		- continuous data: average of the nearest neighbor
		- categorical data: most commonly occurred value
	- inconsistent values
- Duplicate data:
- Documentation that goes along with the data

## The preprocessing steps
- Aggregation
- Sampling
- Simensionality reduction
- Feature subset selection
- Feature creation
- Discretization and binarization
- Variable transformation

### Aggregation
### Sampling
Why? Because for data miners, it is too expensive or time consuming to process all the data.

Q: is the sample representative?

keys:
- sample size
- sampling techniques
	- Simple random sampling
		- where equal numbers of objects are drawn
		- sampling without replacement
		- sampling with replacement
	- Stratified sampling
		- where the number of objects drawn from each group is proportional to the size of that group
	- Progressive sampling
		- it starts with a samll sample, and then increase the sample size until a sample of sufficient size has been obtained. 

### Dimensionality reduction
1. the curse of dimensionality

defn: many types of data analysis become significantly harder as the dimensionality of the data increases. As dimensionality increases, the data becomes increasingly sparse in the space that it occupies.

- for classification: this can mean that there are not enough data objects to allow the creation of a model that reliably assigns a class to all possible objects.
- for clustering, the definitions of density and the distance between points become less meaningful.

2. Dimensionality reduction techniques based on linear algebra approaches
PCA, SVD

### Feature subset selection
- Embedded approaches
- Filter approaches
- Wrapper approaches

### Feature creation
- Feature extration
- Mapping the data to a new space
	- Fourier transform to the time series
- Feature construction
	- Q: are we still keeping the old features

### Discretization and Binarization
Discretization: transform a continuous feature to a categorical feature

Discretization is typically applied to features that are used in classification or association analysis.
- unsupervised discretization
	- equal width approach
	- equal frequency approach
	- K-means
	- more methods
- supervised discretization
	- entropy-based approach

Binarization: transform both continuous and discrete features to one or more binary feature 

Example:
old data = (awful,poor,OK,good,great)
new data = ((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))
### Variable transformation

## Measure of similarity and dissimilarity
similarity and dissimilarity are used by a number of data mining techniques, such as clustering, nearest neighbor classification, and anomaly detection.

Attribute Type | Dissimilarity | Similarity
------------- | -------------- | -----------
Nominal | d = 0 if x = y | s = 1 if x = y |
~ |  d = 1 if x != y | s = 0 if x = y |
Oridinal | d = abs(x-y)/(n-1) | s = 1 - d|
~ | values mapped to integers 0 to n-1, where n is the number of values | ~|
Interval or Ratio | d = abs(x-y) | s = decreasing function (d) |

proximity is used to refer to either similarity or dissimilarity. Proximity measures include: 
- Correlation 
- Euclidean distance
	- L1 norm
	- L2 norm
	- L\infty
	- Properties
		- Positivity
		- Symmetry
		- Triangle Inequality

- Similarity measures
	- Simple matching coefficient
	- Jaccard coefficient
	- Cosine similarity
	- Extended Jaccard Coefficient

Q: Issues in proximity calculation:

1. how to handle the case in which attributes have different scales and/or are correlated,
	- standardization for different scales
	- the Mahalanobis distance when correlation exists
2. how to calculate proximity between objects that are composed of different types of attributes, e.g., quantitative and qualitative,
	- algorithm for similarities of heterogeneous objects
		1. For the k^{th} feature, compute a similarity, s_k(x,y), in the range [0,1];
		2.  define an indicator variable, I_k, for the k^{th} feature as follows: I_k = 0 if the k^{th} feature is an asymmetric feature and both objects have a value of 0, or if one of the objects has a missing value for the k^{th} feature; I_k = 1 if otherwise.;
		3.  compute the overall similarity between the two objects using the following formula:
			- similarity (x,y) = SUM [w_{k} * I_{k} * s_{k}(x,y)]/SUM [I_{k}]
3. and how to handle proximity calculation when attributes have different weights; i.e., when not all attributes contribute equally to the proximity of objects
	- use the weights w

Select the right proximity measure:
- for continuous features: distance measure,
- for sparse data, which often consists of asymmetric features: similarity measures that ignore 0-0 match,

# Chapter 3 Exporing Data (Exploratory Data Analysis)
- Summary statistics
- Visualization
- On-Line Analytical Processing (OLAP)
	- it consists of a set of techniques for exploring multidimensional arrays of values
	- focus on ways to create summary data tables from a multidimensional data array 

## The Iris Data Set
```{r}
library(ggplot2)
data1=read.table("E:/Data/iris.txt",header=F,sep = ",",fill = TRUE)
names(data1)=c('SepalLength','SepalWidth','PetalLength','PetalWidth','class')
summary(data1)
ggplot(data1,aes(PetalLength,PetalWidth,colour=class))+
        # geom_line() +
        geom_point()
```

## Summary Statistics
It captures the characteristics of data set with a single numbers or numbers.

- Mean/Median
	- the median is the middle value if the number of values is odd, 
	- the median is the average of the two middle values if the number of values is even.
	- Mean is sensitive to skewness and outliers.
		- The trimmed mean
- Variance/Standard deviation
	- Variance is sensitive to outliers, so the other available measures are:
		- absolute average deviation, 
		- the median absolute deviation, 
		- the interquartile range (IQR).
- Range
- Frequency
- Mode: the value with the highest frequency
- Percentile
- Skewness

## Visualization
- weather
- the economy
- the political elections

## Techniques
- Visualization of a small number of attributes
	- Stem and Leaf plot
	- Histogram
	- Box plot
	- Scatter plot
		- It graphically shows the relationship between two attributes: linear, non-linear
		- It investigates the degree to which two features separate the classes.
- Visualization of data with spatial/temporal attributes
	- Contour plot
	- Surface plot
	- Vector plot
	- Lower-dimensional slices/Animation
		- e.g. Four dimension data
- Visualization of data with many features
	- Matrix
		- a data matrix can be visualized as an image by associating each entry of the data matrix with a pixel in the image. The brightness or color of the pixel is determined by the value of the corresponding entry of the matrix.
	- Parallel coordinates
	- Star coordinates and chernoff faces 

## OLAP and multidimensional data analysis
we can aggregate or split the data

# Chapter 4 Classification: Basic Concepts, Decision Trees, and Model Evaluation
Classification is learning a function that maps x to the predefined class labels y.

- decision tree
- rule-based
- neural networks
- support vector machine
- naive Bayes classifier

Consusion matrix
Performance metric
- accuracy
- error rate
- true positive rate
- false positive rate

## Decision Tree Induction
### How it works?
We solve the classification problem by asking questions about the features. Each time we receive an answer, a follow-up question is asked until we reach a conclusion.

The series of questions are organized in the form of a decision tree, which is a hierarchical structure consisting of nodes and directed edges. The three types of nodes in the tree are:
- root node
	- no incoming edges
- internal node
	- exactly one incoming edge and two or more outgoing edges
- leaf or terminal node
	- exactly one incoming edge and no outgoing edges

![DecisionTreeDemo](http://i.imgur.com/9CEu82P.jpg)

## Hunt's Algorithm
It is the basis of many existing decision tree induction algs, including ID3, C4.5, and CART.

In Hunt's algorithm, a decision tree is grown in a recursive fashion by partitioning the training records into successively purer subsets. Let D_{t} be the set of training records that are associated with node t.

	1. If all the records in D_{t} belong to the same class label y_{t}, then the node t is a leaf node labeled as y_{t}.
	2. If all the records in D_{t} contain more than one class, an **feature test condition** 
	 is selected to partition the records into smaller subsets. A child node is created for each outcome of the test.
	3. repeat Step 1.

![Hunt's alg demo](http://i.imgur.com/0qHy9dv.jpg)

Q: how to choose the feature test condition? how to evaluate the method?

- Binary features
- Nomial features:
	- ![Norminal features multiway splits](http://i.imgur.com/mhJpf3c.jpg)
- Ordinal features: 
	- the split should preserve the order among the feature values
- Continuous features:
	- ![test condition for continuous feature](http://i.imgur.com/15RZ30F.jpg)

Let p_{i} denote the fraction of records belonging to class i in the node. The measures for selecting the best split are often based on the degree of purity of the nodes.

- Entropy
- Gini
- Classification error

The three measures are consistency.

Test of the goodness of a split:
1. The difference between the degree of purity of the parent node and that of the child node. 
	- The difference = Purity in parent - Weighted average impurity measure in child
	- The larger the difference (the gain), the better the test condition.

2. The gain ratio (in C4.5)
3. Restrict the feature test conditions to binary splits only (in CART)

Q: When to stop the tree-growing process?

## Algorithm for Decision Tree Induction

Example: use a decision tree classifier to distinguish between accesses by human users and those by web robots.

## Characteristics of Decision Tree Induction
1. Decision tree induction is a nonparametric approach for building classification models. In other words, it does not require any prior assumptions regarding the type of probability distribution satisfied by the class and other features. 
2. Finding an optimal decision tree is an NP-complete problem.
3. Techniques developed for constructing decision trees are computationally inexpensive, making it possible to quickly construct models even when the training set size is very large.
4. Decision trees are relatively easy to interpret. The accuracies of the trees are also comparable to other classification techniques.
5. Decision trees are quite robust to the presence of noise.
6. The presence of redundant features does not adversely affect the accuracy of decision trees.
7. Data fragmentation problem. at the leaf nodes, the number of records may be too small to make a statistically significant decision about the class label. One possible solution is to disallow further splitting when the number of records falls below a certain threshold.
8. Subtree replication problem
9. Decision boundary problem. The test condition so far only involves a single feature at a time. That means, the decision boundaries are parallel to the coordinate axes. It can not classify the situation when the decision boundary is oblique.

## Model Overfitting
errors include:
- training errors
- prediction errors (test errors tend to be larger than training errors)

### Causes for overfitting
- Present of Noise
- Lack of Representative Samples
- Multiple Comparison 
	- As the alternative/comparison number, k, increases, the chance of finding a "good" candidate increases, unless the threshold is also modified to account for k.

### How to estimate the prediction error?
- training error
- pessimistic error: training error + a penalty term for model complexity
- minimum description length
- use a validation set

### Handling overfitting in decision tree induction
- prepruning
- post-pruning

## Comparing a classifier
- Holdout method
- Random subsampling
- k-fold cross-validation
- Bootstrap
	- sampling with replacement
- Model comparison
	- hypothesis test

# Chapter 5 Classification: Alternative Techniques  
- Rule-based classifier
- Nearest-neighbor classifier
- Support vector machine
- Ensemble methods

## Rule-Based Classifier
It uses a collection of "if ... then ..." rules
![equation](http://www.sciweavers.org/tex2img.php?eq=R%3D%28r_1%20%5Cvee%20r_2%20%5Cvee%20...r_k%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
where ri, the classification rule, can be expressed in the following way.
![equation](http://www.sciweavers.org/tex2img.php?eq=%20r_i%3A%20%28Condition_i%29%20%5Crightarrow%20y_i%0A%0Awhere%20Condition_i%20%3D%20%28A_1%7Eop%20%7E%5Cnu_1%29%5Cvee%20%28A2%7Eop%20%7E%5Cnu_2%29%5Cvee%20...&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
(the logical op is upsidedown here)

The left-hand side of the rule is called the rule antecedent or precondition where op is a logical operator. The right-hand side of the rule is called the rule consequent.

### How a rule-based classifier works
- Mutually exclusive rules
- Exhaustive rules
- Ordered rules
- Unordered rules

### How to build a rule-based classifier
- direct methods
	- extract classification rules directly from data
	- sequential covering algorithm
	- RIPPER algorithm
- indirect methods
	- extract classification rules from other classification models, such as decision trees and neural networks. For example, every path from the root node to the leaf node of a decision tree can be expressed as a classification rule.

### Rule evaluation

### Characteristics of Rule-Based Classifiers
1. The expressiveness of a rule set is almost equivalent to that of a decision tree because a decision tree can be represented by a set of mutually exclusive and exhaustive rules.
2.

## Nearest-Neighbor Classifiers
A nearest-neighbor classifier represents each example as a data point in a d-dim space, where d is the number of features. The k-nearest neighbors of a given example z refer to the k points that are closest to z. The data point is classified based on the majority class labels of its neighbors.
![KNN](http://i.imgur.com/LTXaWvV.jpg)

### Q: How to choose the value k?
If k is too small, overfitting, if k is too large, biased. One possible solution is to weight the influence of x according to its distance.

### Characteristics of Nearest-Neighbor Classifiers
1. NN classification is part of a more general technique known as instance-based learning, which requires a proximity measure to determine the similarity or distance between instances and a classification function that returns the predicted class of a test instance based on its proximity to other instances.
2. Lazy learners such as NN classifiers do not require model building. However, classifying a test example can be quite expensive because we need to compute the proximity values individually between the test and training examples. In contrast, eager learners often spend the bulk of their computing resources for model building. Once a model has been built, classifying a test example is extremely fast.
3. NN classifier make decisions based on local infomation.
4. NN classifier can produce flexible decision boundaries.
5. NN classifier is sensitive to scales of features.

## Bayesian Classifiers
- Naive Bayes
- the Bayesian belief network
It is an approach for modeling probabilistic relationships between the feature set and the class variable. The posterior probability P(Y|X), if P(Yes|X) > P(No|X), then the text data is classified as Yes.

### Naive Bayes Classifer
Assumption: the features are conditionally independent.
![equation](http://www.sciweavers.org/tex2img.php?eq=%20P%28Y%7CX%29%3D%5Cfrac%7BP%28X%7CY%29P%28Y%29%7D%7BP%28X%29%7D%20%3D%20%5Cfrac%7B%20%5Cprod_i%5Ed%20P%28X_i%7CY%29P%28Y%29%7D%7BP%28X%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

### If the training examples do not cover many of the attribute values, we may not be able to classify some of the test records because the P will be 0. One possible solution is to use the m-estimate approach for estimating the conditional probabilities:


![equation](<http://www.sciweavers.org/tex2img.php?eq=P%28x_i%7Cy_j%29%3D%20%5Cfrac%7Bn_c%7D%7Bn%7D%20%0A%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
where n_c could be small or 0
![equation](http://www.sciweavers.org/tex2img.php?eq=P%28x_i%7Cy_j%29%3D%20%5Cfrac%7Bn_c%2Bmp%7D%7Bn%2Bm%7D%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0)
where m is a parameter known as the equivalent sample size, and p is the prior probability of observing the feature value x among records with class y.

### Characteristics of Naive Bayes Classifiers
- Naive Bayes classifiers are robust to isolated noise points because such points are averaged out when estimating conditional probabilities from data. 
- They can also handle missing values by ignoring the example during model building and classification.
- They are robust to irrelevant features. If x is an irrelevant feature, then P(x|y) becomes almost uniformly distributed. (But why?) The class-conditional probability for x has no impact on the overall computation of the posterior probability. 
- Correlated features can degrade the performance of Naive Bayes classifiers.

### Bayes Error Rate

### Bayesian Belief Networks (BBN)
The conditional independence assumption made by naive Bayes classifiers may seem too rigid. Instead of requiring all the attributes to be conditionally independent given the class, the Bayesian Belief Networks allows us to specify which pair of features are conditionally independent. BBN provides a graphical representation of the probabilistic relationships among a set of variables. There are two key elements of a Bayesian network:
- a directed acyclic graph (dag) encoding the dependence relationships among a set of variables.
- a probability table associating each node to its immediate parent nodes.

![BBN](http://i.imgur.com/MKpy0s8.jpg)

Properties
1. (Conditional Independence) A node in a Bayesian network is conditionally independent of its non-descendants, if its parents are known.

![BBN with tables](http://i.imgur.com/5B9T48W.jpg)

### Model Building
- create the structure of the network
- estimate the probability values in the table associated with each node.

Once the right topology has been found, the probability table associated with each node is determined. 

## Artificial Neural Network (ANN)
- Perceptron
- Multilayer Artificial Neural Network

### Perceptron
- input nodes
- output nodes
	- the sign function acts as an activation function for the output neuron, outputs a value +1 if its argument is positive and -1 if its argument is negative.
- the weighted link
	- training a perceptron model amounts to adjusting the weights of the links until they fit the input-output relationships of the underlying data.
	- ![equation](http://www.sciweavers.org/tex2img.php?eq=w_j%5E%7Bk%2B1%7D%3Dw_j%5E%7Bk%7D%20%2B%20%20%5Clambda%20%28y_i-%5Chat%20y_i%5E%7Bk%7D%29x_%7Bij%7D%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
	- the parameter, learning rate, whose value is between 0 and 1, can be used to control the amount of adjustments made in each iteration. If lamda is close to 0, then the new weight is mostly influenced by the value of the old weight. On the other hand, if lamda is close to 1, then the new weight is sensitive to the amount of adjustment performed in the current iteration.
	- If the problem is not linearly separable, the algorithm fails to converge.
![Perceptron](http://i.imgur.com/QFE9A6P.jpg)

### Multilayer Artificial Neural Network
The perceptron is a single-layer, feed-forward neural network.
- Input layer
- Hidden layer
- Output layer
	- Various types of activation functions, include sign function, linear, sigmoid, and hyperbolic tangent functions.

![Multilayer ANN](http://i.imgur.com/2wYKHmt.jpg)
![activation functions](http://i.imgur.com/84MmnRH.jpg)

Intuitively, we can think of each hidden node as a perceptron that tries to construct one of the two hyperplanes, while the output node simply combines the results of the perceptrons to yield the decision boundary.
![Multilayer ANN2](http://i.imgur.com/rFcRb5T.jpg)

Q: How to learn the ANN model?
A: Gradient descent method.
![equation](http://www.sciweavers.org/tex2img.php?eq=w_j%20%5Cleftarrow%20w_j%20-%20%20%5Clambda%20%20%5Cfrac%7BError%20Func%28w%29%7D%7Bw_j%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

For hidden nodes, it is difficult to assess the error term. A technique known as back-propagation is used:
- the forward phase
	- the weights at level k+1 are updated after the weights at level k
- the backward phase
	- the weights at level k+1 are updated after the weights at level k+2

### Characteristics of ANN
- the number of nodes in the input layer should be determined. Assign an input node to each numerical or binary input variable. If the input variable is categorical, we could either create one node for each categorical value or encode the k-ary variable using log2_k input nodes.
- the number of nodes in the output layer should be established. For a k-class problem, there are k output nodes.
- The number of hidden layers and hidden nodes, and feed-forward or recurrent network architecture should be established. 
- Training examples with missing values should be removed or replaced with most likely values.
- Multilayer neural networks with at least one hidden layer are universal approximators; i.e., they can be used to approximate any target functions.
- ANN are sensitive to the presence of noise in the training data. One possible approach to handle noise is to use a validation set to determine the generalization error of the model.
- Training an ANN is a time consuming. Nevertheless, test examples can be classified rapidly.

## Support Vector Machine (SVM)
SVM works with handwritten digit recognition, text categorization, high-dimentsional data and avoids the curse of dimensionality problem.

Support vector: a subset of the training examples that represents the decision boundary.

To illustrate the basic idea behind SVM:
- maximal margin hyperplane
- a linear SVM in linearly separable data
- extend SVM to non-linearly separable data

### Maximal Margin Hyperplane
Each decision boundary B is associated with a pair of hyperplanes, which are obtained by moving a parallel hyperplane away from the decision boundary until it touches the closet point. The distance between these two hyperplanes is the margin of the classifier.
![Margin hyperplanes](http://i.imgur.com/QhUnFDM.jpg)

#### Rationale for Maximum Margin
Decision boundaries with large margins tend to have better generalization errors than those with small margins. Classifiers that produce decision boundaries with small margins are more susceptible to model overfitting and tend to generalize poorly on previously unseen examples.

**Structural risk minimization (SRM)**, which is another way to express generalization error as a tradeoff between training error and model complexity. SRM says that there is an upper bound to the generalization error of a classifier (R) with a probability of 1-\eta:
![equation](<http://www.sciweavers.org/tex2img.php?eq=E_%7Btest%7D%20%5Cleq%20%20E_%7Btraining%7D%20%2B%20%5Cpsi%20%28%20%5Cfrac%7Bh%7D%7BN%7D%2C%20%20%5Cfrac%7Blog%28%20%5Ceta%20%29%7D%7BN%7D%20%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
where \psi is a monotone increasing function of h, h is the model complexity, N is the training size.

To minimize the worst-case test error, the model complexity h has to be reduced, thus, the margin is maximized. The margin is inversely related to the model complexity h. Models with small margins have higher complexity because they are more flexible and can fit many training sets. 

### Linear SVM: Separable Case (Maximal Margin Classifier)
A linear SVM is a classifier that searches for a hyperplane with the largest margin.

#### Linear Decision Boundary
![equation](http://www.sciweavers.org/tex2img.php?eq=y%20%3D%5Cbegin%7Bcases%7D1%20%26%20w%20x%20%2B%20b%20%3E%200%5C%5C-1%20%26%20wx%20%2B%20b%20%3C%200%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![Decision Boundary](http://i.imgur.com/HGE5iza.jpg)

#### Margin of a Linear Classifier
The margin is the distance between these two hyperplanes. 
![equation](http://www.sciweavers.org/tex2img.php?eq=d%20%3D%20%20%5Cfrac%7B2%7D%7B%7C%7Cw%7C%7C%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

#### Learning a Linear SVM Model
The training phase of SVM involves estimating the parameters **w** and **b** of the decision boundary such that 
![equation](http://www.sciweavers.org/tex2img.php?eq=y%20%3D%5Cbegin%7Bcases%7D1%20%26%20w%20x%20%2B%20b%20%3E%200%5C%5C-1%20%26%20wx%20%2B%20b%20%3C%200%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

as well as maximize the margin:
![](http://i.imgur.com/Hbr4qEu.jpg)

This turns out to be a quadratic programming problem:
![](http://i.imgur.com/2zkWxDA.jpg)

To solve the problem, use the Lagrange multiplier method. What we do is to rewrite the objective function in a form that takes into account the constraints imposed on its solutions. The new objective function is known as the Lagrangian for the optimization problem: 
![](http://i.imgur.com/D8d9ydx.jpg)
where \lamda is called the Lagrange multipliers. Assume \lamda > 0, it is clear that any feasible solution may only decrease the value of the Lagrangian. In other words, any infeasible solution will only increase the value of the Lagrangian.

One way to handle the inequality constraints is to transform them into a set of equality constraints. 
![KKT](http://i.imgur.com/CsRcjMI.jpg)
The constraints states that the Lagrange multiplier \lamda must be zero unless the training data x satisfies the equation
![](http://i.imgur.com/VtKgRJJ.jpg)
Such training data lies along the hyperplanes and is known as a **support vector**.

Take the derivative of Lp with respect to w and b and set them to 0. Plug them back to the Lp, we will have the dual formation of the optimization problem (dual problem). The dual Lagrangian involves only the Lagrange multipliers and the training data, while the primary Lagrangian involves the Lagrange multipliers as well as parameters of the decision boundary. 

The dual optimization problem can be solved using numerical techniques such as quadratic programming. Once the \lamda is found, use the \lamda to find the parameters w and b. The decision boundary can be expressed as follows:
![SVM decision boundary](http://i.imgur.com/2ln8rHZ.jpg)
Once the decision boundaries are found, a test function z is classified.

### Linear SVM: Nonseparable Case (Soft Margin)
The SVM formulation presented in the previous section constructs only decision boundaries that are mistake-free. This section examines how the formulation can be modified to learn a decision boundary that is tolerable to small training errors using a method known as the **soft margin** approach, where the learning algorithm in SVM must consider the trade-off between the width of the margin and the number of training errors committed by the linear decision boundary.
![SVM non separable case](http://i.imgur.com/keLfjBo.jpg)

The constraints are modified by introducing **positive-valued** slack variable (\xi), as shown in the following:
![SVM slack variables](http://i.imgur.com/m4aQyyQ.jpg)
The slack variable provides an estimate of the error of the decision boundary.
The objective function must be modified to penalize a decision boundary with large values of slack variables, given by:
![svm modified objective function](http://i.imgur.com/MxrMoGT.jpg)
where C and k can be chosen based on the model's performance on the validation set.

### Nonlinear SVM
The trick here is to transform the data from its original coordinate space in x into a new space f(x). One potential problem is the curse of dimensionality problem associated with high-dimensional data. 

#### Kernel trick
The kernel trick is a method for computing similarity in the transformed space using the original feature set. The dot product in the transformed space can be expressed in terms of a similarity function in the original space:
![SVM kernel func](http://i.imgur.com/oJntK0B.jpg)
The similarity function, K, is the kernel function.
- We do not have to knoe the transformation function. K satisfies Mercer's theorem which says that the K can always be expressed as the dot product between two input vectors in some high-dimensional space. The transformed space of the SVM kernels is called a reproducing kernel Hilbert space (RKHS). 
- Computations performed in the original space avoid the issues associated with the curse of dimensionality problem.

### Characteristics of SVM
- The SVM learning problem can be formulated as a convex optimization problem, in which efficient algorithms are available to find the global minimum of the objective function. Other classification methods tend to find only locally optimum solutions.
- SVM maximizes the margin of the decision boundary and the soft margin.
- SVM can be applied to categorical data.

## Ensemble Methods
The classification methods present so far predict the class labels using a single classifier induced from the training data. The ensemble methods improve classification accuracy by aggregating the predictions of multiple classifiers. The ensemble methods constructs a set of base classifiers and performs classification by taking a majority vote on the predictions made by each base classifier. 
![Ensemble methods](http://i.imgur.com/BTzKNEA.jpg)
### Why the ensemble methods are better?
The ensemble methods work better with unstable classifiers, i.e., decision trees, rule-based classifiers, and artificial neural networks.

### Constructing an Ensemble classifier
- By manipulating the training set
	- Bagging
	- Boosting
- By manipulating the input features
	- a subset of input features is chosen for each training set
	- good for data set with redundant features
	- Random forest, which uses decision trees as its base classifiers
- By manipulating the class labels
	- when the number of class labels is large
- By manipulating the learning algorithm

### Bias-Variance Decomposition
It is for analyzing the prediction error.

- Bias
- Variance
- Noise
	- when the target is not stationary

### Bagging (Bootstrap aggregating)
It samples from a data set with replacement according to a uniform probability distribution.
Bagging improves test error by reducing the variance of the base classifiers.

- If a base classifier is unstable, bagging helps to reduce the errors associated with random fluctuations in the training data.
- If a base classifier is stable, i.e., robust to minor perturbations in the training set, then the error of the ensemble is primarily caused by bias in the base classifier, where bagging may not be able to improve the performance of the base classifiers significantly.

### Boosting
Boosting is an iterative method used to adaptively change the distribution of training data so that the base classifiers will focus on examples that are hard to classify. Boosting assigns a weight to each training data and may adaptively change the weight at the end of each boosting round. The weights can be used:

- as a sampling distribution to draw a set of bootstrap samples from the original data
- by the base classifier to learn a model that is biased toward higher-weight samples.

####Algorithm example
	1. The data examples are assigned equal weight. i.e., they are equally likely to be chosen for training.
	2. a classifier is induced from the training set from step 1
	3. Update the weights. Examples that are classified incorrectly will have their weights
	   increased, while those that are classified correctly will have their weights decreased.
	4. Repeat
	5. Ensemble the base classifiers obtained.

Over years, there are many implementations of the boosting algorithm. They differ in terms of:
- How the weights are updated at the end of each boosting round
- How the predictions made by combining those classifiers

Cons: because of its tendency to focus on training example that are wrongly classified, the boosting technique can be quite susceptible to overfitting.
#### AdaBoost
The weight update mechanism for AdaBoost is given by the equation:
![AdaBoost](http://i.imgur.com/LruDFZy.jpg)
where w_i^j denotes the weight assigned to example data x_i,y_i during the jth boosting round, e_i denotes the error rate of each base classifier i.

### Random Forest
Random forest combines the predictions made by multiple decision trees, where each tree is generated based on the values of an independent set of random feature vectors. The random feature vectors are generated from a fixed probability distribution. Bagging using decision trees is a special case of random forests. 
Randomization helps to reduce the correlation among decision trees so that the generalization error of the ensemble can be improved.
![Random forest](http://i.imgur.com/OKxNw8g.jpg)
Q: How to generate the random feature vectors from a fixed probability distribution?
- Forest-RI (random input selection)
	1. Randomly select F input features to split at each node of the decision tree.
	2. Grow the tree to its entirety without any pruning
	3. Predictions are combined using a majority voting scheme.
The number of features chosen F = log_2_d + 1, where d is the number of input features. If F is small, then the trees tend to become less correlated, on the other hand, the strength of the tree classifier tends to decrease.
- Forest-RC
	1. Create linear combinations of the input features when d is too small. Specifically, at each node, F of new combined features, each of them is linearly combined using a uniform distribution in the range of [-1,1] from L randomly selected input features.

The classification accuracies of random forest are quite comparable to the AdaBoost algorithm, but more robust to noise and runs much faster than the AdaBoost algorithm.

The table below shows the empirical results obtained when comparing the performance of a decision tree classifier against bagging, boosting, and random forest. The base classifiers used in each ensemble method consist of fifty decision trees. The classification accuracies reported in the table are obtained from ten-fold cross-validation. Notice that the ensemble classifiers generally outperform a single decision tree classifier on many of the data sets.

### Class Imbalance Problem
Examples: credit card fraud detection, defective product detection

This section presents some of the methods developed for handling the class imbalance problem.
- alternative metrics besides accuracy are introduced, along with a graphical method called ROC analysis
- Cost-sensitive learning and sampling -based methods
#### Alternative Metrics
The accuracy measure treats every class as equally important, it may not be suitable for analyzing imbalanced data sets, where the rare class is considered more interesting than the majority class. 

Confusion matrix
- precision
	- precision (p) measures the fraction of predictions that actually turns out to be positive
- recall
	- recall (r) measures the fraction of positive examples correctly predicted 

It is challenge to build a model that maximizes both precision and recall
- arithmetric mean
- geometric mean
- harmonic mean (F1), which is closer to the smaller of the two numbers

### the Receiver Operating Characteristic Curve (ROC)
It is a graphical approach for displaying the tradeoff between true positive rate (y) and false positive rate (x) of a classifier. The ideal model is y=1, x = 0.

A good classification model should be located as close as possible to the upper left corner of the diagram, while a model that makes random guesses should reside along the main diagonal. 
![roc](http://i.imgur.com/eO8HW0Y.jpg)

### Cost-sensitive learning

### Sampling-Based Approaches
The idea of sampling is to modify the distribution of instances so that the rare class is well represented in the training set, including undersampling, oversampling, and a hybrid of both approaches.

### Multiclass Problem
- One-against-rest (1-r) approach
	- decomposes the multiclass problem into K binary problems.
- One-against-one (1-1) approach
	- constructs K(K-1)/2 binary classifiers, where each classifier is used to distinguish between a pair of classes.
- The error-correcting output coding (ECOC) method
	- for multiclass learning, each class is represented by a unique bit string of length n known as its codeword. The predicted class of a test instance is given by the codeword whose Hamming distance is closest to the codeword produced by the binary classifiers. Recall that the Hamming distance between a pair of bit strings is given by the number of bits that differ.
	- **Multiclass learning** requires that the row-wise and column-wise distances of the codewords must be well separated. A larger column-wise distance ensures that the binary classifiers are mutually independent, which is an important requirement for ensemble learning methods.

# Chapter 8 Cluster Analysis: Basic Concepts and Algorithms
Cluster analysis divides data into groups (clusters) that are meaningful, useful, or both. 
- Clustering: divide objects into groups (**detect patterns**)
- Classification: assign particular objects to groups

### Clustering for understanding
- Biology: apply cluster to analyze the large amounts of genetic information, to find groups of genes that have similar functions
- Information retrieval: group search results into a small number of clusters. For example, a query of "movie" might return web pages grouped into categories such as reviews, trailers, stars, and theaters.
- Climate: find patterns in the atmophere and ocean
- Psychology and Medicine: detect patterns in the spatial or temporal distribution of a disease
- Business: identify segment customers

### Clustering for utility: find the most representative cluster prototypes
- apply to a reduced data set consisting only of cluster prototypes, instead of applying the algorithm to the entire data set.
- compression (vector quantization) 
- efficiently find nearest neighbors

Clustering algorithms:
- K-means
- agglomerative hierarchical clustering
- DBSCAN
- advanced algorithms

## Overview
- What is cluster analysis?
- The relationship to other techniques that group data
- Different ways to group a set of objects into a set of clusters
- types of clusters

### What is cluster analysis?
cluster analysis groups data based on data. The goal is that the objects within a group be similar to one another and different from the objects in other groups

### Different types of clusterings
- Hierarchical vs. Partitional
	- Partitional clustering: a division of data into non-overlapping clusters
	- Hierarchical clustering: 
- Exclusive vs. Overlapping vs. Fuzzy
	- Exclusive
	- Overlapping: a data can belong to more than one group
	- Fuzzy clustering: every object belongs to every cluster with a membership weight that is between 0 and 1
- Complete vs. Partial
	- a complete clustering assigns every object to a cluster, whereas a partial clustering does not.

### Different types of clusters
- well-separated
- prototype-based (center-based cluster)
	- continuous case: centroid, i.e., the mean of all the points in the cluster
	- discrete case: medoid, i.e., the most representative point of a cluster
- graph-based
- density-based
- shared-property
![different types of clusters](http://i.imgur.com/Cxmokvn.jpg)

- K-means
	- this is a prototype-based, partitional clustering technique that attempts to find a user-specific number of clusters (K), which are represented by their centroids.
- Agglomerative Hierarchical Clustering
	- this is a collection of closely related clustering technique that produces a hierarchical clustering by starting with each point as a singleton cluster and then repeatedly merging the two closest clusters until a single, all encompassing cluster remains.
- DBSCAN
	- this is a density-based clustering algorithm that produces a partitional clustering in which the number of clusters is automatically determined by the algorithm.

## K-means
### Algorithms

### Evaluating the algorithms
We consider each of the steps in the algorithm in more detail and then provide an analysis of the algorithm's space and time complexity.

For step 1, the point is: once we have specified a proximity measure and an objective function, the centroid that we should choose can often be determined mathematically. For example, when the Euclidean distance and SSE are applied, 

Data in Euclidean space: Euclidean distance
Document data: Jaccard measure
- the objective is to max the similarity of the documents in a cluster to the cluster centroid, which is known as the cohesion of the cluster.
Euclidean data: Manhattan distance as the proximity function where the centroid is median

#### Choose initial centroids
- Random Initialization
- Take a sample of points and cluster them using a hierarchical clustering technique. K clusters are extracted from the hierarchical clustering, and the centroids of those clusters are used as the initial centroids.
- Bisecting K-means
	- To obtain K clusters, split the set of all points into two clusters, select one of these clusters to split, and so on, until K clusters have been produced.
- use postprocessing to fixup the set of clusters produced

#### Time and Space complexity
The time requirements for K-means are basically linear in the number of data points. In particular, the time required is O(I*K*n*(p+1)).

The space requirement is O((n+K)(p+1)), where m is the number of points and p+1 is the number of features.

#### Handling empty clusters
One possible approach is to choose the replacement centroid from the cluster that has the highest SSE.

#### Outliers
With the squared error criterion is used, outliers can unduly influence the clusters that are found

#### Reduce the SSE
we want to improve the SSE, but do not want to increase the number of clusters (By increasing K, we can reduce the SSE).

Two possible ways to increse K are:
1. Split a cluster with largest SSE
2. Introduce a new cluster centroid

Two possible ways to decrese K are:
1. Disperse a cluster by removing the centroid and reassign the points to other clusters.
2. Merge two clusters

#### Update centroids
Instead of updating cluster centroids after all points have been assigned to a cluster, the centroids can be updated incrementally, after each assignment of a point to a cluster.

How to choose the target cluster to split?
One possible way is to choose the one with the largest SSE.

#### Strength and weaknesses
K-means has difficulty detecting the natural clusters when clusters have non-spherical shapes or widely different sizes or densities. This is because the objective function in the k-means is a mismatch for the kinds of clusters we are trying to find since it is minimized by globular clusters of equal size and density. However, this problem can be overcome if subclusters are permitted.

K-means also has trouble in clustering data that contains outliers.

Finally, K-means is restricted to data for whhich there is a notion of a center.
![](http://i.imgur.com/7rYad65.jpg)
![](http://i.imgur.com/Iq3FyJG.jpg)
![](http://i.imgur.com/HfOkNeK.jpg)

## Agglomerative Hierarchical clustering
There ar two basic approaches for generating a hierarchical clustering:
- Agglomerative
	- start with the points as individual clusters and, at each step, merge the closest pair of clusters. This requires defining a notion of cluster proximity.
- Divisive
	- start with one, all-inclusive cluster and, at each step, split a cluster until only singleton clusters of individual points remain. In this case, we need to decide which cluster to split at each step and how to do the splitting

A hierarchical clustering is often displayed graphically using dendrogram or a nested cluster diagram.
![Hierarchical clustering](http://i.imgur.com/zYWA7Qk.jpg)

#### Algorithm
starting with individual points as clusters, successively merge the two closest clusters until only one cluster remains.

**Defining proximity between clusters**
It is the definition of cluster proximity that differentiates the various agglomerative hierarchical techniques.
- MIN (single link): the cluster proximity as the proximity between the closet two points that are in different subsets of nodes.
	- 
- MAX (complete link or CLIQUE): the cluster proximity as the proximity between the farthest two points in different clusters to be the cluster proximity, or using graph terms, the longest edge between two nodes in different subsets of nodes
- Group average: the proximity as the average pairwise proximities of all pairs of points from different clusters.
![cluster proximity](http://i.imgur.com/5SF2Vbt.jpg)

**Time and Space complexity**
The time requirements for basic agglomerative hierarchical clustering algorithm are basically O(m^3).
The space requirement for basic agglomerative hierarchical clustering algorithm is basically O(m^2) since the algorithm uses a (symmetric) proximity matrix (n^2/2) + the space needed to keep track of the clusters (n-1).

