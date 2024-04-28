---
title: "Task Background for session 2"
teaching: 15
exercises: 30
questions:
- "What is our goal?"
- "Which data are we going to use?"
objectives:
- "Understand the task goal and obtain the relevant data"
keypoints:
- "Ask questions, experiment, and help others"
- "Everyone is here to learn and that means making mistakes"
---
# Machine learning task

Machine learning is a type of AI in which we don't explicitly code our algorithms, but instead teach them by showing examples of input and output, and letting the algorithms learn the relation between them.
The algorithms that we use have already been constructed to be of general use and to be able to learn, it is up to us to tach them wrong from right.

There are many different machine learning algorithms, many classes of machine learning, and different ways to train an ML.
The art of machine learning, of being a good data scientist, is to be able to take a given data set and desired outcome and select and train an appropriate algorirthm.
In reality there is no super magic to this and we engage in a little bit of 'try many things and choose the best'.

The normal machine learning workflow is as follows:

![ML workflow]({{page.root}}{% link fig/ML_workflow.jpeg %})
*credit:[medium.com](https://medium.com/@maloojinesh/machine-learning-for-beginners-from-zero-level-8be5b89bf77c)*

In order that we can complete our lesson within the given time we are going to work with some pre-made data (skipping steps 1+2), and train only a single model.
We'll leave the improvements (step 5) for future work.


## Our task

In this lesson our aim is to use historical data to predict the value of future data.
Specifically we'll be predicting future sunspot activity.

As can be seen below there is a very long history of tracking these data, that there is some cyclic behaviour, and that there are a few deviations to the "usual" cycles.
All up this makes for a very interesting data set.
We are going to focus on a forecasting the sunspot numbers up to 1 year in advance, and we'll only be using data from after the Dalton Minimum. 

![Sunspot numbers]({{page.root}}{% link fig/Sunspot_Numbers.png %})
*credit:[wikipedia](https://upload.wikimedia.org/wikipedia/commons/2/28/Sunspot_Numbers.png)*

## Our data set

The file we are looking for is a record of average Sunspot counts per month since 1749.
The data has been compiled, calibrated, and aggregated already, and is available on [kaggle](https://www.kaggle.com/datasets/robervalt/sunspots/data).
To download the datset we could either register an account on Kaggle or use the kaggle python module.

> ## Use kaggle to download the data
> ~~~
> kaggle datasets download -d robervalt/sunspots
> unzip sunspots.zip
> ~~~
> {: .language-bash}
{: .challenge}


Inspect the data:

~~~
,Date,Monthly Mean Total Sunspot Number
0,1749-01-31,96.7
1,1749-02-28,104.3
2,1749-03-31,116.7
3,1749-04-30,92.8
4,1749-05-31,141.7
5,1749-06-30,139.2
6,1749-07-31,158.0
7,1749-08-31,110.5
8,1749-09-30,126.5
9,1749-10-31,125.8
10,1749-11-30,264.3
...
3254,2020-03-31,1.5
3255,2020-04-30,5.2
3256,2020-05-31,0.2
3257,2020-06-30,5.8
3258,2020-07-31,6.1
3259,2020-08-31,7.5
3260,2020-09-30,0.6
3261,2020-10-31,14.4
3262,2020-11-30,34.0
3263,2020-12-31,21.8
3264,2021-01-31,10.4
~~~
{: .output}

The first (un-named) column is the row index.

Steps:
- Load data using numpy
- Create time/counts data sets
- Decompose the data into trend/season/residual
- Work on trend data (trim as required)


## ML methods

Machine learning has a number of classificaitons.
Today we'll be employing supervised learning to train a regression model.
**Supervised learning** means that we have examples of right and wrong answers to test our trained models against.
**Regression** means that we'll be predicting numerical answers for some given input.
The particular method that we'll be using is called **random forest**, which is a collection of randomly generated **descision trees**.
Furthermore, we'll be working with **time series** data, meaning data which has an implicit order and often has internal peridicity and correlations.

## A brief intro to ML

In machine learning we refer to our input data as 'features' and our output as 'predictions'.
Not all of the input data is useful, so part of the role of a machine learning practitioner is to choose appropriate features.

The model that we'll be using today is based on a **decsision tree**.
A descision tree can be thought of as a long string of `if/else` statements that are used to chose the relevant output.
In the case of a classification task, the output is a class or label.
In a regression task the output is numeric and so the final part of the `if/else` tree is a linear model based on one of the input features.

In fact, most ML algorithms are, at thier heart, a bunch of linear models.
These linear models can reproduce out and capture behaviours that are non-linear.

An example of a decision tree predicting classes based on three features is shown below:

![Descision tree]({{page.root}}{% link fig/DescisionTree.png%})
*credit:[scikitlearn](https://scikit-learn.org/stable/modules/tree.html)*

Descision trees are easy to understand, east to implement, and are able to be analysed after being trained to gain insight into the relation between the features and the predictions.
However, descision trees work by making `if/else` choices that partition the feature space along boundaries perpendicular to the feature axies.
This makes it hard for them to capture non-linear relationships or make predictions that are correlated with a linear combination of inputs.
To make such predictions the trees need to be very deep, which takes a long time to train them.

Rather than building a single tree which captures all the realtionships within our data (building a single expert), we can instead rely on the wisdowm of the crowd by building many simpler trees and then taking an average of their predictions.
We can thus build a foest of trees.
The focus here is on building a large number of simple trees, thus instead of allowing the trees to work with all the available features, only a subset are presented.

![Random Forest]({{page.root}}{% link fig/RandomForest.png %})
*credit:[gitconnected.com](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84)*


TODO find the nice images of the auto-regression tables

## Working with our data

There are a veriety of different cycles in our data.
We can separate out some of these cycles using a decomposition technique:

> ## Decompose our data
> TODO
{: .challenge}

Select only the data after the Dalton Minimum (1830 onward).

## Training, testing, and validation

Split our data into training and testing data (80/20).
Split our training data into train / validate subsets.

If we are teaching our ML machine we need to be careful not let it peak at the answers.
To do this we need to split our data into two sets, one for training and one for evaluating the effectiveness on unseen data.
For this we'll make a training a testing split of 80/20%.

However, in order to train our algorithm we need to do a few iterations of the train/evaluate loop.
For this we can split our training data further into a training and validation set.
In fact, we do this many times for different non-overlapping choices of the validation set.

The splitting scheme is shown in the figure below:

![Cross validation]({{page.root}}{% link fig/CVDiagram.png %})
*credit: [learningds.org](https://learningds.org/ch/16/ms_cv.html)*

> # Make test/train subsets of our data
> Since timeeries data has inherent order and internal correlation we cannot
> just choose a random 80/20 split.
> Instead we are going to choose the first/last 80/20 % for the splitting.
> TODO 80/20 split
{: .challenge}

We will work on the validation part at a later stage since it'll require some more careful thinking.

We do the test / train split now becaue we don't want any information leaking from our test data into our model in the form of the choices that we are making about the data processing.

## Selecting features

The simplest features that we have access to are our data points but with a lag.
We could also crate new features based on combinations of these lags, differences, averages, etc.
For now we'll just focus on the the lags.

Potentially we have hundreds or thusands of lags we could select from, but the more features that we select, the more chance that we'll be adding noise to our model.
This is refered to as the curse of dimensionality: too few features and we have not enough signal, too many features and we have too much noise.
There is a sweet spot somewhere between the two, and it's often at a much smaller number that you might think.

![Curse of dimensionality]({{page.root}}{% link fig/CurseOfDimensionality.png %})
* Credit: [builtin.com](https://builtin.com/data-science/curse-dimensionality)*

Because we are working with linear models, and directly with lagged data, the lags that are most likely to be useful are those that correspond to periodicities in the dataset.
Let us now investigate what those peridicities are.
Since we know that the solar cycle has a period of around 11 years (132 months) we'll look at all the periods in the data up to at least 132 months and select the strongest.

~~~
# Autocorrelation and partial correlation plots for up to 150 lags.
~~~
{: .language-python}

- feature selection
    - linear model based on history
    - look for periodicity = correlation between features (lags)


## Training our model

TODO: Select the model with some default parameters and train it
~~~

~~~
{: .language-python}

TODO: Evaluate the model and make some observations

~~~
# predicted values vs known values

# model accuracy measures
~~~
{: .language-python}


> ## Discuss the performance so far
>
{: .discussion}

## Tuning our model

TODO Hyperparameter training.

TODO choosing validation subsets.
