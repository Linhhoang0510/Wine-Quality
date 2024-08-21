# Wine-Quality

A wide range of measures can affect the quality of wine, for example, grape types, wine 
brands, climates, and so on. In general, the wine quality is evaluated in two parts. One 
part measures physicochemical features, like alcohol level, pH value, residual sugar, 
acidity levels, etc., for further analysis and production. At the same time, the other one is 
sensory tests which are mainly carried out by wine specialists. However, the taste of 
experienced experts is subjective, and the procedures are often complex, costly, and 
time-consuming, while different people will have divergent senses of what constitutes 
good or bad quality. To solve this problem, machine learning algorithms are applied to find 
a model to predict the wine quality based on various wine physicochemical 
measurements, providing an objective perspective to the traditional wine assessment. 

The dataset used for this paper is the Wine Quality obtained from the UCI Machine 
Learning Repository. It consists of two separate datasets:
- Red wine (1599 samples)
- White wine (4898 samples)
which are variants of the Portuguese "Vinho Verde" wine.

Besides, there are a total of 12 variables in each dataset, with input variables including fixed acidity, 
volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, 
density, pH, sulphates, and alcohol, and quality is the target variable.
The true quality values are based on sensory tests, ranging from 0 to 10, where higher 
scores indicate better quality.

2 machine learning approaches were implemented to predict wine quality. The first method is the Decision Tree algorithm which 
involves building a decision tree - a tree-shaped graph to visualize and model decision 
paths. Another approach explores the capabilities of using neural networks in learning 
complex data patterns and constructing and training a deep learning model for wine 
quality prediction.

Model Workflows:

![Wine Quality drawio (1)](https://github.com/user-attachments/assets/4650a2d4-bd97-4641-9788-162e47d6a275)
