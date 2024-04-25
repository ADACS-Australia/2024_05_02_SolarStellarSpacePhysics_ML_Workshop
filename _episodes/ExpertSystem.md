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


![AI, ML, and DL]({{page.root}}{% link fig/AI_ML_DL.png %})
<!-- Credit to https://medium.com/analytics-vidhya/ai-ml-dl-whats-what-ecb354967e63 -->

In this workshop we will design and old school class of AI (Artificial Intelligence) - an expert system.
This system will be designed to replicate the analysis process conducted by an expert who is familiar with the data and the task at hand.
As such, we'll first be doing some work manually to understand what features of the images are important to the task, and then doing some computing work to design a system that replicates our HI (Human Intelligence).

We will be focusing on determining:

1. If a given image contains a signal of interest
2. Whereabouts in that image ths signal is located

We will then discuss how we can turn the above *detection* algorithm into one which also *characterises* the signals of interest.

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
> # builtins
> import os
> 
> # 3rd party libraries
> from astropy.io import fits
> 
> import matplotlib
> from matplotlib import pyplot as plt
> 
> import numpy as np
> from numpy import random
> 
> import pandas as pd
> 
> import scipy
> from scipy.stats import zscore
> ~~~
> {: .language-python}
{: .challenge}


> ## ImportError
> If you get an import error for any of the above libraries it means that you haven't installed the library yet.
> The libraries required for this session are outlined in the [Setup]({{page.root}}{% link _episodes/Setup.md %}) lesson.
>
> You can install libraries directly from your jupyter notebook with a code cell like this:
> ~~~
> !pip install numpy
> ~~~
> {: .language-python}
> 
{: .caution}


## Copy the data that we'll be using

In this session we'll be skipping the data collection and preparation stage and instead download some ready made data.

> ## Download the data
> Download and unzip the following file into a directory called 'data'. Link: TODO
>
> You should see a bunch of `.fits` images with names like:
> ~~~
> TD_20110730-M9_3.fits
> TD_20111225-M4_0.fits
> TD_20120305-M2_1.fits
> TD_20120705-M1_6.fits
> TD_20121023-X1_8.fits
> TD_20130113-M1_0.fits
> TD_20130422-M1_0.fits
> TD_20130817-M3_3.fits
> ~~~
> {: .output}
>
> If you have trouble downloading or unzipping the data please raise a hand or put up a red sticker.
{: .challenge}

An example of the data that we'll be working with is shown below:

![Time distance plot]({{page.root}}{% link fig/TD_clear.png %})

### Load and view our data

In order to work with the image data we'll need to load it and view it in our notebook.

> ## Load and view an image
> 1. Select the image 'TD_20110730-M9_3.fits'
> 2. Load this image using the `fits.open()` function and assign it to the variable `hdu`
> 3. Extract the image data using `data = hdu[0].data`
> 4. Use the following code snippet to view the image:
> 
> ~~~
> fig, ax = plt.subplots(figsize=(15,9))
> ax.imshow(data, origin='lower')
> ax.set_xlabel('Distance')
> ax.set_ylabel('Time')
> plt.show()
> ~~~
> {: .language-python}
> 
> > ## Solution
> > ~~~
> > hdu = fits.open('data/TD_20110730-M9_3.fits')
> > data = hdu[0].data
> > ~~~
> > {: .language-python}
> > ![Raw data]({{page.root}}{% link fig/RawData.png %})
> {: .solution}
{: .challenge}




## Detection vs Characterization

- Is there something here?
- What does that something look like?

## Manual inspection

Before we can make some AI we have to understand how our HI works.
Let's start by looking at the following three images which have a combination of signal and noise.
The signal that we are looking for is strong or medium in the left and center image, and is not present in the right image.

| Strong | Medium | None |
| -- | -- | -- |
| ![Strong]({{page.root}}{% link fig/TD_clear.png %}) | ![Medium]({{page.root}}{% link fig/TD_medium.png %}) | ![None]({{page.root}}{% link fig/TD_none.png %}) |


> ## What is the "feature"?
> Inspect the image above and discuss the following with your fellow learners:
> 1. What is common in all three images and how do you describe it?
> 2. What features are in the first two images and not the third?
> 3. How are the things that you describe in (1) and (2) different from each other?
> 
> After you have had some discussion, summarize your notes in the [etherpad]({{site.ether_pad}}).
> 
{: .discussion}


## Algorithm planning

Now that we have identified the "feature" (a ripple) in the images above we need to design an algorithm to detect said feature.

> ## A few things to note about the images:
> - The ripple is a curved arc which looks to start around 20-30 time units from distance 0
> - Most of the pixels in the image are noise pixels, with only a small fraction being signal pixels
> - The background image is filled with waves that are primarily horizontally elongated
> - The background waves and the ripple are **harder** to distinguish at large Distances because they are aligned
> - The background waves and the ripple are **easier** to separate at distances around 10-10Mm as the two are misaligned 
> - The "wave height" or strength of the waves is different in different parts of the image, with the left of the image sometimes being min/max on the colour scale.
> 
{: .solution}

The above observations are going to help us in designing an algorithm which separates the signal of interest (the ripple) from the noise (the other wavy things).
The first few steps that we can plan are:
- Normalise the data set so that the noise is roughly equal over the image
- Crop the image to show just the region where the signal is the strongest
- Estimate the curvature of the ripple feature by over-plotting the (cropped) image


### Normalize the data

Our data is stored in a `numpy` array so we can use the `data.std(axis=?)` function to determine the standard deviation of the data.
Here we can set `axis=0` to do the calculation along the time axis so we see the std vs distance.
~~~
distance_std = data.std(axis=0)
plt.plot(distance_std)
plt.title("STD per distance slice")
plt.show()
~~~
{: .language-python}

> ## Compute the std along the distance axis
> Copy the above code into a notebook cell and execute to see the STD vs distance plot.
> Then make a new cell which will do the same thing but this time using `axis=1` so that we see the STD vs time.
> 
{: .challenge}

> ## STD plots
> ![STD vs Distance]({{page.root}}{% link fig/STDvsDist.png %}) 
> ![STD vs Time]({{page.root}}{% link fig/STDvsTime.png %})
> 
{: .solution}

Finally, we can removed this variation from the images by subtracting the mean and dividing by the standard deviation.
To do this we'll use the following from scipy:

~~~
scaled_data = zscore(data)
~~~
{: .language-python}

If we now replot our image and add a colour bar we get something like the following:
~~~
fig, ax = plt.subplots(figsize=(15,9))
im = ax.imshow(scaled_data, origin='lower', vmin=-3, vmax=3)
fig.colorbar(im, ax=ax, label="SNR")
ax.set_xlabel('Distance')
ax.set_ylabel('Time')
plt.show()
~~~
{: .language-python}

![Scaled image]({{page.root}}{% link fig/ScaledData.png %})

In the above image the color scale has been cropped to $\pm 3\sigma$.
Compared to the original image we have:
- Reduced the prominence of whatever is going on at time $\sim 20$ and distance $\sim 0$
- Increased the prominence of the ripple in the regin around distance $\sim 50$


At this point we can make a new observation - the signal seems to follow a $t \sim \sqrt{d}$ relation starting at $d=0$ and $t=\sim25$.

Since our signal is distributed over multiple pixels, we could sum along the path of the signal, and hopefully the signal will accumulate while the noise will cancel out (regression to the mean).
This is a standard approach and relies on the signal having some coherence over the summation whilst the noise does not.
Since the signal is only really visible in some of the plot we shouldn't do this sum over the entire arc of the ripple, thus we'll crop our image first.

## Crop image and identify the path of the ripple

Let's first crop the image as follows:

~~~
cropped_data = scaled_data[:,25:75]

fig, ax = plt.subplots(figsize=(15,9))
im = ax.imshow(cropped_data, origin='lower', vmin=-3, vmax=3)
fig.colorbar(im, ax=ax, label="SNR")
ax.set_xlabel('Distance -25 units')
ax.set_ylabel('Time')
plt.savefig("Cropped.png")
plt.show()
~~~
{: .language-python}

![Cropped]({{page.root}}{% link fig/Cropped.png %})

> ## Parameterise the t vs d relation
> Using the following snippet as a starting point, determine the relationship between time and distance (in pixel coordinates).
> ~~~
> #Add lines +/-2.5 pix around the feature for highlighting
> distance = np.arange(cropped_data.shape[1])
> ax.plot(distance, <my guess function> +2.5, lw=2, color='red', label='Feature of interest')
> ax.plot(distance, <my guess function> -2.5, lw=2, color='red')
> ~~~
> {: .language-python}
>
> Once you have a relationship between t and d post your best result in the [etherpad]({{site.etherpad}}).
> 
{: .challenge}

My example is below:

![CroppedAnnotated]({{page.root}}{% link fig/CroppedAnnotated.png %})

> ## My function
> ~~~
> times = 5*(distance+20)**0.5+22
> ~~~
> {: .language-python}
{: .solution}


Now that we have a description of what the relationship is we can sum along that line.
Unfortunately our data are organized in orthogonal axes of time and distance but we want to look along a path with is some combination of the two.
We can get around this problem by reprojecting our data so that the signal of interest is parallel to one of our axes.
To do this we take our function from above and shuffle each of the columns downward so that our curved ripple becomes a horizontal line.
In this process we'll be shuffling the entire image which means that the current horizontal waves which are the noise will become curved.
Thus if we want to sum along our path we can use `np.sum(axis=?)` and get a nice profile of our potential signal.

### Reproject the data and aggregate

We'll shuffle the data along the time axis, one slice at a time, using the `np.roll()` function.
This will take care of all the boundary problems that we might have.

~~~
times =  # your function from above
offsets = np.int64(times)  # Convert to integers so we can use as an index

rolled = cropped_data.copy()  # copy data incase we make a boo-boo
mid = cropped_data.shape[0]//2  # determine the mid point of the vertical axis
for i in range(rolled.shape[1]):
    rolled[:,i]=np.roll(rolled[:,i],
                 -offsets[i] + mid)  # Cause our signal to lie in the middle of our plot
~~~
{: .language-python}


With the above implemented we should see something like the below:

![Reprojected]({{page.root}}{% link fig/Reprojected.png %})

Now if we sun across the distance dimension we should be able to accumulate the signal and wash out the noise.

~~~
summed = rolled.sum(axis=1)  # Sum over distance

# plot
fig, ax = plt.subplots(figsize=(15,9))
ax.plot(summed)

# add lines to draw attention
ax.axvline(mid+5, lw=2, color='red')
ax.axvline(mid-5, lw=2, color='red')

ax.set_xlabel('Wared Time')
ax.set_ylabel("Sum over distance")
ax.set_title("Summed data")
plt.show()
~~~
{: .language-python}

![Summed data]({{page.root}}{% link fig/Summed.png %})

Within the red bars above we see a farily impressive signal that is different from anything outside the red bars.

We can determine the location of this potential signal by looking at the max value and location:

~~~
peak_val = np.max(summed)
peak_index = np.argmax(summed)
print(f"Peak of {peak_val:5.2f} found at {peak_index}")
~~~
{: .language-python}

> ## What next?
> We could use a matched filter. TODO
{: .challenge}

## Creating a signal detection metric

What we would really like to have is a single metric that we can use to determine if there was a detection or not.
With some modification to our above method we can do this.

For example, we are not looking to characterise our signal, so it's strength is un important to us onlt it's presence.
We have already normalised the data set once, so there isn't any harm in repeating this again.

Instead of using the sum of our data we can use the mean instead:

~~~
d_stat = np.mean(rolled, axis=1)
~~~
{: .language-python}


## Recap and consolidation

Let us recap what we have done so far in the form of some functions that will allow us to reproduce our work easily:

> ## Our functions
> ~~~
> def read_scaled_data(filename: str):
>     """
>     Read and scale data from the given filename.
>     
>     Parameters
>     ----------
>     filename : str
>         File to open
>         
>     Returns
>     -------
>     hdr, data : fitsheader, np.ndarray
>         The fits header and the image data.
>     """
>     img = fits.open(filename)
>     data = zscore(img[0].data)
>     return img[0].header, data
> 
> 
> def crop_data(img, start=25, step=50):
>     """
>     Crop the image data using default options
>     
>     Parameters
>     ----------
>     start, step : int
>         The start point of the crop and the size
>         
>     Returns
>     -------
>     start : int
>         The offset into the array for the crop
>         
>     data : n.ndarray
>         The cropped data
>     """
>     return start, img[:,start:start+step]
> 
> 
> def get_time_offsets(distances, zeropoint):
>     """
>     Calculate the time offsets according to our relation
>     
>     parameters
>     ----------
>     distances : np.array
>         An array of distances (indexes)
>         
>     zeropoint : int
>         An offset for our function in the case that we are working
>         with cropped data
>         
>     returns
>     -------
>     times : np.array
>         An array (of ints) which are the time offsets
>     """
>     times = 5*(distances+zeropoint)**0.5
>     return np.int64(times)
> 
> 
> def roll_data(data, offsets):
>     """
>     Shuffle the data according to the given offsets to (hopefully)
>     make the signal of interest more prominent
>     
>     parameters
>     ----------
>     data : np.ndarray
>         2d image data
>         
>     offsets : np.array
>         An array of ints which are the offsets to be applied
>         
>     returns
>     -------
>     rolled : np.ndarray
>         The shuffled data
>     """
>     rolled = data.copy()
>     for i in range(rolled.shape[1]):
>         rolled[:,i]=np.roll(rolled[:,i], -offsets[i])
>     return rolled
> ~~~
> {: .language-python}
>
{: .solution}

With our above functions we can then look at the detection metric for **all** the files in our data directory.

> ## Combine the whole workflow together into a single function
> Combine all the above functions together to complete the following function
>
> ~~~
> def process_file(fname):
>     """
>     Compute our detection metric on the given file
>     
>     parameters
>     ----------
>     fname : str
>         Filename to load
>         
>     returns
>     -------
>     zeropoint : int
>         The offset into the dataset that we are 
>         
>     best_stat : float
>         The maximum of the detection statistic
>     """
>     hdr, data = read_scaled_data(fname)
> 
>     ...
> 
>     return zeropoint, best_stat
> ~~~
> {: .language-python}
>
> > ## Suggested solution
> > ~~~
> > def process_file(fname):
> >     """
> >     ...
> >     """
> >     hdr, data = read_scaled_data(fname)
> > 
> >     zeropoint, cropped_data = slice_data(data)
> > 
> >     distances = np.arange(data.shape[1])
> > 
> >     t_offsets = get_time_offsets(distances,zeropoint)
> > 
> >     if flip:
> >         cropped_data = cropped_data[::-1,:]
> >     cropped_data = roll_data(cropped_data, t_offsets)
> > 
> >     d_stat = np.mean(cropped_data, axis=1)
> > 
> >     best_stat=np.max(d_stat)
> >     return zeropoint, best_stat
> > ~~~
> > {: .language-python}
>  {: .solution}
{: .challenge}

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
