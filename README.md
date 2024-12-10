Exploring Decision Trees in Machine Learning 
Student Name: R. Meda 
Student ID: 23005003 
Introduction 
Decision trees are a fundamental tool in predictive modelling because they are among the most 
straightforward and understandable machine-learning methods. They can create a "tree-like" structure to 
generate predictions by breaking data into subgroups according to feature values. Because of their efficacy 
and intelligibility, they are frequently utilized in tasks like regression and classification. Decision trees are also 
crucial to comprehend in the larger context of machine learning as they serve as the foundation for more 
complex algorithms like Random Forests and Gradient Boosting. 
Using the Iris dataset as a real-world example, this lesson examines the foundations of decision trees, their 
use, and performance tweaking. Along with a practical Python implementation, it also discusses the benefits, 
drawbacks, and applications of decision trees. 
Understanding Decision Trees 
A decision tree works by recursively splitting data into subsets, aiming to create "branches" that end at 
"leaves." Each leaf represents a final decision or prediction. The splits are chosen to maximize the purity of 
the resulting subsets, measured using metrics like: 
• Entropy and Information Gain: Focuses on reducing uncertainty by measuring the "disorder" in the 
data. 
• Gini Impurity: Calculates the likelihood of incorrect classification for a randomly chosen element. 
Decision trees are highly flexible, capable of modelling non-linear relationships, and can handle both 
numerical and categorical data. However, this flexibility comes at a cost: without proper tuning, decision trees 
can easily overfit the training data, reducing their ability to generalize. 
Key parameters to tune: 
1. Max Depth: Restricts the number of splits, controlling overfitting. 
2. Min Samples Split: Sets the minimum number of samples required to split a node. 
3. Criterion: Defines the metric used to measure split quality, such as Gini or Entropy. 
 
Dataset 
The Iris dataset is a classic in statistics and machine learning, introduced by Ronald A. Fisher in 1936. It has 
since become a benchmark for evaluating classification algorithms. The dataset consists of 150 samples of 
iris flowers divided into three species: 
1. Setosa: Easily distinguishable due to its unique feature patterns. 
2. Versicolour: Slightly overlaps with Virginica, making it harder to classify. 
3. Virginica: Shares some feature characteristics with Versicolour, presenting a non-linear decision 
boundary. 
Each sample is described using four numerical features: 
• Sepal Length (cm): Length of the outer petal-like structure. 
• Sepal Width (cm): Width of the outer petal-like structure. 
• Petal Length (cm): Length of the inner petal structure. 
• Petal Width (cm): Width of the inner petal structure. 
The balanced nature of the dataset, with 50 samples per class, and its simple structure make it ideal for 
demonstrating decision trees. 
This table provides an overview of features used to classify flowers in a dataset, such as the famous Iris 
dataset. It includes five attributes: Sepal Length and Sepal Width, which measure the dimensions of the 
Dataset Structure 
flower's sepal in centimeters, and Petal Length and Petal Width, which provide similar measurements for the 
petals. These numerical attributes help in distinguishing the physical characteristics of flowers. The final 
attribute, "Species," represents the class label (e.g., Setosa) that categorizes the type of flower. An example 
row illustrates a sample flower with specific feature values, where the species is identified as "Setosa." 
 
Code Implementation 
To understand the separability of the classes, visualizations such as histograms and scatter plots of Petal 
Length and Petal Width can be created. These features tend to separate Setosa clearly, while Versicolour 
and Virginica overlap. 
 
To avoid overfitting, this method limits the depth to three while training a Decision Tree Classifier using the 
Iris dataset. The model is assessed using accuracy, a classification report, and a confusion matrix, with Gini 
impurity serving as the split criteria. Lastly, the tree is shown to show how iris species are categorized using 
characteristics. 
Analysis of Iris Dataset 
1. Accuracy: The decision tree achieves high accuracy on the Iris dataset due to its simplicity and 
separable feature patterns. 
2. Confusion Matrix: Highlights perfect classification of Setosa and minor overlaps between Versicolour 
and Virginica. 
3. Tree Visualization: Displays how the tree splits based on features like Petal Length and Petal Width. 
 
Advantages of Decision Trees with Reference to the Iris Dataset 
The Iris dataset showcases the strengths of decision trees in handling simple yet structured data. Here’s how 
decision trees excel: 
1. Simplicity and Interpretability 
The Iris dataset's small size and clear feature separability align perfectly with decision trees' intuitive nature. 
For example, the tree splits at Petal Length ≤ 2.5 to easily classify Setosa while showing clear branching 
logic for other species. This visual representation is ideal for explaining decision-making processes to both 
technical and non-technical audiences. 
2. Handles Non-linear Relationships 
In cases like Versicolour and Virginica, where feature overlaps exist, decision trees create non-linear splits 
such as Petal Width ≤ 1.8 to adapt to the complexity, without requiring feature transformations. 
3. No Need for Feature Scaling 
Decision trees directly process features like Petal Length (ranging from 1.0 to over 6.0) and Petal Width (0.1 
to 2.5), eliminating the need for scaling, which is especially useful in datasets like Iris. 
4. Works with Numerical Features 
The entirely numerical nature of the Iris dataset allows decision trees to split data efficiently without requiring 
additional preprocessing. Features like Petal Length are prioritized naturally for accurate predictions. 
5. Feature Importance 
Decision trees rank features like Petal Length and Petal Width as the most influential for separating species, 
providing insights into which features contribute most to classification. 
 
Limitations of Decision Trees with Iris Dataset Examples 
Decision trees, though highly interpretable and effective, exhibit notable limitations, even when applied to the 
Iris dataset: 
1. Overfitting: Decision trees, when left unrestricted, tend to overfit training data by memorizing 
intricate patterns, including outliers. In the Iris dataset, this could lead to excessive complexity where 
specific data points dominate the model. Regularization parameters like max_depth or 
min_samples_split help control this tendency. 
2. Overlapping Classes: Certain classes in the Iris dataset, particularly Versicolour and Virginica, 
have overlapping feature values such as Petal Width. Decision trees generate rigid boundaries, 
which may result in misclassifications for data points lying near these boundaries. 
3. Instability: Decision trees are sensitive to variations in the dataset. Even minor changes, such as 
the addition or removal of a few samples, can drastically alter the structure of the tree, affecting its 
reliability and consistency. 
4. Scalability: While decision trees handle small datasets like Iris efficiently, their performance 
diminishes with larger datasets. They can become computationally intensive, leading to slower 
processing and less effective generalization when the feature space or sample size increases. 
Real-World Applications Using Iris Dataset as Inspiration 
The simplicity and versatility of the Iris dataset serve as an excellent foundation for applying decision trees 
to solve real-world problems: 
1. Healthcare Diagnostics: Decision trees can classify patients into various risk categories (e.g., low, 
moderate, high risk) based on diagnostic features such as age, symptoms, or test results, like how 
Iris species are classified using petal and sepal measurements. 
2. Fraud Detection: Decision trees analyse transactional features like amount, frequency, and location 
to determine whether transactions are "legitimate" or "fraudulent." This process parallels 
distinguishing between overlapping Iris species like Versicolour and Virginica. 
3. Customer Segmentation: Businesses use decision trees to group customers based on behaviours 
(e.g., spending habits or engagement levels). For instance, they can classify customers as "high
value" or "low-value," drawing parallels to the way decision trees classify Iris species. 
4. Agricultural Predictions: Decision trees predict crop types or yields based on input features such 
as soil pH, temperature, and moisture levels. This process mirrors how the Iris dataset is used to 
classify flowers based on their measurable traits. 
5. Environmental Monitoring: Decision trees help predict air or water quality categories (e.g., Good, 
Moderate, Poor) by analysing environmental features like particulate matter, temperature, or humidity. 
This is conceptually similar to using petal and sepal dimensions to classify Iris species. 
References 
1. Iris Dataset - UCI Machine Learning Repository 
One of the most well-known datasets in statistics and machine learning is the Iris dataset. It was first 
presented by Ronald A. Fisher in 1936 and has since been widely used as a standard by which to 
measure classification methods. The dataset is perfect for teaching and testing basic machine
learning methods like decision trees because of its balance, simplicity, and well-separated classes. 
Readers may immediately get the dataset for their studies using this reference. 
 
2. GitHub Repository - Machine-Learning 
The Iris dataset is among the most well-known in machine learning and statistics. Since Ronald A. 
Fisher initially proposed it in 1936, it has been widely accepted as a benchmark for evaluating 
classification techniques. The dataset's balance, simplicity, and well-separated classes make it ideal 
for teaching and testing fundamental machine-learning techniques like decision trees. With this 
reference, readers may obtain the dataset for their research right now. 
 
3. "Elements of Statistical Learning" - Hastie, Tibshirani, Friedman. 
With its thorough explanations of algorithms including decision trees, Random Forests, and boosting 
techniques, this book is a mainstay of the machine learning literature. It clarifies ideas that are 
essential to comprehending decision trees, such as information gain, entropy, and Gini impurity. Citing 
this book emphasizes how theoretically sound your paper is. 
 
4. Python Libraries - Matplotlib, Seaborn. 
The report's visuals, which included scatter plots and histograms, were made using these Python 
modules. Understanding the structure of the dataset and the classifier's performance requires the use 
of visualizations. By connecting these libraries, readers may access resources for project 
documentation and tutorials. 
Conclusion 
The capabilities of decision trees, such as their capacity to manage non-linear connections, rank significant 
attributes, and provide understandable models, are demonstrated by the Iris dataset. Although there are 
drawbacks, such as overfitting and sensitivity to data changes, tuning can lessen them. Decision trees 
continue to be adaptable instruments with uses in healthcare, finance, agriculture, and other fields, providing 
a balance between ease of use and effectiveness in resolving practical issues.
