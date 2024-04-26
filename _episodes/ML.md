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

## Download the required data

The file we are looking for is a record of Sunspot counts per month since 1749.

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
- Discuss our goal
    - predicting trend data 1 yr in advance
- Discuss methods
    - Regression
    - Measuring success
    - Decision trees
    - Random forests
    - Timeseries predictors
- Start with conditioning our data
    - normalisation
    - looking for missing data (there is none)
    - split into train/test sets
        - take 80/20 train test
- feature selection
    - linear model based on history
    - look for periodicity = correlation between features (lags)
- train, test, and validation discussion
- split train into train/validate

