---
title: "Task Background for session 1"
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

## Task background

Observations with the Solar Dynamics Observatory ([SDO](https://sdo.gsfc.nasa.gov/)) Helioseismic and Magnetic Imager ([HMI](http://hmi.stanford.edu/)) are used to create ... TODO


Our task is to search these images and look for wave fronts which represent ... TODO

In this workshop we will design and old school class of AI (Artificial Intelligence) - an expert system.
This system will be designed to replicate the analysis process conducted by an expert who is familiar with the data and the task at hand.
As such, we'll first be doing some work manually to understand what features of the images are important to the task, and then doing some computing work to design a system that replicates our HI (Human Intelligence).

We will be focusing on determining:

1. If a given image contains a signal of interest
2. Whereabouts in that image ths signal is located

We will then discuss how we can turn the above *detection* algorithm into one which also *characterises* the signals of interest.

### Detection vs Characterisaion

- Is there something here?
- What does that something look like?

## Download the required data

- Data on google drive
- link to data
- show .png files of the images for people who don't have ds9 or similar on their machine
- download a file with the expert analysis for each image

## Manual inspection

- look at examples images which have a strong signal, weak signal, and no signal
- choose a file in which you think the signal is particularly strong
- choose a file in which you think there is not signal at all

## Algorithm planning

- What features of the image did you use to determine if our signal is present?
- How do you describe the signal of interest?
- How do you describe the remainder of the data?
- How could we separate the two?

Signal looks like $t \sim \sqrt{d}$.

Seems to be no signal in the low d range, at higher d the signal is harder to separate from the noise.
Around 25-75 distance units we see the best contrast between signal and noise.

Summing along the path of the signal should allow us to accumulate signal whilst the noise will cancel out (regression to the mean).


## Notebook setup

- Start a notebook
- Import libraries that we'll need later
- Load an image and plot it

## Enhancing the signal

- zscale normalisation of the image
- cropping the image to the region of highest contrast
- reprojecting the image so that the signal of interest is aligned with our coordinate axes
- sum along our coordinate axes
- boom, signal is there (or not)

## Creating a signal detection metric

- max of spectrum?
- matched filter to our signal profile?

## Determining signal vs noise

- making images with no signal via flipping
- using images with no known signal (thanks to expert)
- measuring our detection metric
- set threshold of noise +/- sigma

## Measure the effectiveness of our detection algorithm

- comparison with expert system
- precision, recall, confusion matrix
- the ROC

## Future work

- higher resolution images
- interpolating to make the re-projection more accurate (avoid round offs)
- additional "trials" of the $t/d$ relation
- larger data set for training
- Using more than a single point of reference in the image (multiple crops)
- building a characterization machine

## Wrap up
