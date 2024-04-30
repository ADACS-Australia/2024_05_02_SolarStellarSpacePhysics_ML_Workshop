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

## Setup our session

We are going to be using Jupyter notebooks to explore our data and design our algorithm.
Once we have an algorithm we are happy with we could then create a standalone program that runs from the command line, but Jupyter notebooks are a much better place for us to work and explore during this phase of development.


You can run Jupyter from the command line via:
~~~
jupyter lab
~~~
{: .language-bash}

Alternatively, you can use [Google colaboratory](https://colab.research.google.com/) to run notebooks directly from your Google Drive.

Finally, you can use the Jupyter extension to [VSCode](https://code.visualstudio.com/) to run Jupyter notebooks directly in your VSCode editor.

> ## Start a new notebook
> Using one of the three methods above, create a new notebook and run the following code to import the libraries that we'll be working with today.
> ~~~
> %matplotlib inline
> import matplotlib
> import matplotlib.pyplot as plt
> 
> import numpy as np
> 
> import pandas as pd
> from pandas.plotting import lag_plot, autocorrelation_plot
> 
> from sklearn.model_selection import TimeSeriesSplit
> from sklearn.metrics import mean_squared_error
> 
> import statsmodels
> import statsmodels.api as sm
> from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
> from statsmodels.tsa.ar_model import AutoReg
> from statsmodels.tsa.stattools import pacf
> ~~~
> {: .language-python}
{: .challenge}


> ## ImportError
> If you get an import error for any of the above libraries it means that you haven't installed the library yet.
> The libraries required for this session are outlined in the [Setup]({{page.root}}{% link _episodes/Setup.md %}) lesson and in ([requirements.txt]({{page.root}}{% link data/requirements.txt%})).
>
> You can install libraries directly from your jupyter notebook with a code cell like this:
> ~~~
> !pip install numpy
> ~~~
> {: .language-python}
> 
{: .caution}

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


> ## Read data with pandas
> 1. Use `pd.read_csv` to read our data using the column names of `index`, `date`, and `y`.
> 2. Drop the `index` column as we won't be working with it.
> 3. Convert the `date` column from `string` into `datetime` format and use it as our table index.
> 4. Remove the `date` column as we won't need it anymore.
> 5. Set the dataframe frequency to be months ('M')
> 6. Crop out all the data before 1820
>
> > ## My solution
> > ~~~
> > def load_data():
> >     # load our data and set up our data frame to be easily used
> >     data = pd.read_csv("Sunspots.csv", names=['index','date','y'], header=0)
> >     data = data.drop(columns='index') # drop the row numbers
> >     data.index = pd.to_datetime(data['date'])  # Convert the date column into datetime object and use it as our index
> >     data.drop(columns='date', inplace=True)  # Delete the date column
> >     data = data.asfreq('M')  # Tell pandas that the data have a cadence of months
> >     data = data['1820':]  # use only data after the Dalton Minimum
> >     return data
> > 
> > data = load_data()
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Run a cell with just `data` in it to inspect the data frame.
It should look something like this:
~~~
date           y
1820-01-31   32.0
1820-02-29   44.4
1820-03-31   7.5
1820-04-30   32.3
1820-05-31   48.9
...          ...
2020-09-30   0.6
2020-10-31   14.4
2020-11-30   34.0
2020-12-31   21.8
2021-01-31   10.4
2413 rows × 1 columns
~~~
{: .output}

> ## Use the following plotting template
> We will be plotting our data many times in this lesson so we'll use the following template.
> ~~~
> fig, ax = plt.subplots()
> ax.plot(data)
> ax.set_xlabel('Date')
> ax.set_ylabel('Mean monthly spot counts')
> #plt.savefig('Data.png')
> plt.show()
> ~~~
> {: .language-python}
{: .challenge}

![Data]({{page.root}}{% link fig/Data.png %})

We are going to be using some ML methods to learn from this data and predict new data into the future.
First we'll learn a little about the method we are going to be using.

## ML methods

Machine learning has a number of classifications.
Today we'll be employing supervised learning to train a regression model.
**Supervised learning** means that we have examples of right and wrong answers to test our trained models against.
**Regression** means that we'll be predicting numerical answers for some given input.
The particular method that we will be exploring is called an **autoregressive** model.
Autoregressive models predict the next value (regression) in a sequence by taking measurements from previous inputs in the sequence.
Our data is a time series of a single value so this is a 1D model that we'll be using.

As an example of an autogressive model, suppose we want to predict the next value in a sequence using the previous two values as input.
The previous two values are often referred to as **lags**.
In the figure below we have a prediction model (the $\script{F}$) taking two lagged values ($X_{t-1}, X_{t-2}$) as input (and optionally extra data $\epsilon$ ) to produce the prediction of $X_t$.

![Auto regression]({{page.root}}{% link fig/AutoReg.png %})
*Credit:[10.3390/info14110598](https://doi.org/10.3390/info14110598)*

Note that in the above diagram the prediction for $X_{t+1}$ uses the previous prediction as an input.
This is a key feature of time-series algorithms, and it means that you can easily end up in a situation where the output is oscillating or growing without bound as the algorithm amplifies prediction errors.
One way to avoid this delirious behavior is incorporate additional (exogeneous) data into the prediction.
This is not within the scope of our lesson for today.


## Lags and auto-correlations

As indicated in the above figure our input data is a lagged version of the data we are trying to predict.
If our data had some easily defined periodicity then this would be a simple problem with not much learning to do.
E.g. a two body system can be described in a closed analytical form and there is no real learning to to do, we just supply some initial conditions and we are done.
However, the Sun is not so simple, there are periods within the data that we can see easily by eye, but there is also a lot of weirdness (a.k.a physics) going on that we don't completely understand.

Since we'll be working with lagged data


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

> ## Make test/train subsets of our data
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

Potentially we have hundreds or thousands of lags we could select from, but the more features that we select, the more chance that we'll be adding noise to our model.
This is refered to as the curse of dimensionality: too few features and we have not enough signal, too many features and we have too much noise.
There is a sweet spot somewhere between the two, and it's often at a much smaller number that you might think.

![Curse of dimensionality]({{page.root}}{% link fig/CurseOfDimensionality.png %})
*Credit: [builtin.com](https://builtin.com/data-science/curse-dimensionality)*

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
