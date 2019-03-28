# Seat availability at *La Permanence* coworking spaces #

[La Permanence](https://www.la-permanence.com "La Permanence coworking
space in Paris") offers two coworking spaces in Paris, in *rue du
Fer à Moulin* and *rue d'Alésia*.

<p align="center">
<img src="La-Permanence-screenshot.png" alt="La Permanence" width="450"/>
</p>

## Goals
Our goals are to **model** and **forecast** the number of available
seats at the locations.  

## The data
Both locations are open 24/7 and the number of available seats is
posted in real time on their website.  The data has been collected (at
irregular but frequent intervals) since mid-January 2019.


<p align="center">
<img src="moulin-availability.png" alt="Availability at La Permanence
coworking space in rue du Fer à Moulin" width="1000" height="250"/>
</p>


The data is saved in `attendance.csv` (a misnomer, since it records
the number of *available* seats rather than the number of seats used).
It consists of three columns:  

1. `timestamp`: date and time of collection of data (in the **UTC**
     standard).  The format is `YYYY-MM-DD-hh-mm-ss` with `YYYY=`year,
     `MM=` month, `DD=` day, `hh=` hour, `mm=`minute, `ss=` second    
1. `Moulin`: number of available places at *rue du Fer à Moulin*  
1. `Alésia`: number of available places at *rue d'Alésia*  

The dataset is also [available on
kaggle](https://www.kaggle.com/antoinechoffrut/la-permanence-attendance).  


## Scripts
1. The script `la-permanence-scraping.py` collects the data.   

## Notebooks
1. The Jupyter Notebook `la-permanence-understanding.ipynb` is a very
   brief presentation of the dataset and its format.  
1. The Jupyter notebook `la-permanence-seasonality.ipynb` investigates
   the double seasonality in the data.
1. The Jupyter notebook
   `la-permanence-random-forests-forecasting.ipynb` illustrates the
   effectiveness of using random forests to forecast  the
   availability of seats at the coworking spaces.   The model used in
   this notebook is the result of extensive exploration into which
   features from the data to feed the model and which values to select
   for its parameters.  This is done in other notebooks.  
1. The Jupyter notebook
   `la-permanence-random-forsts-feature-importance.ipynb` examines
   which features to use in the model.  
1. The Jupyter notebook
   `la-permanence-random-forests-parameters.ipynb` examines how the
   parameters in the model effects its performance.  
1. The Jupyter notebook `la-permanence-decision-trees.ipynb` explores
   how the parameters affect the inner structure of decision trees.  

# Findings

Below we incorporate the most salient conclusions of the project as it
moves along.

## Seasonality
The timeseries exhibits seasonality of multiplicity two: there is a
24-hour as well as a 7-day seasonality.  It makes thus sense to take
averages for each day of the week.


<p align="center">
<img src="moulin-average-availability.png" alt="Average availability
at rue du Fer à Moulin"
width="1000"/>
</p>

This clearly shows that each day of the week has a distinct pattern.

We can analyze this in more detail by first performing a 24-hour
seasonal decomposition on the original time-series, as shown in the
figure below.

<p align="center">
<img src="moulin-24-hour-seasonal-decomposition.png" alt="24-hour
seasonal decomposition of the data from rue du Fer à Moulin" width="1000"/>
</p>

Note how the 7-day seasonality is retained in the trend resulting from
this first decomposition.  Thus we perform a second, 7-day seasonal
decomposition on this trend as illustrated in the figure below.

<p align="center"> <img
src="moulin-7-day-on-24-hour-seasonal-decomposition.png" alt="7-day
seasonal decomposition on the trend resulting from 24-hour seasonal
decomposition of the data from rue du Fer à Moulin" width="1000"/>
</p>


## Random forests
We use random forests to predict seat availability at the coworking
spaces for the next two weeks.  

### Choosing the features
Since the data consists of pure time-series, we first need to
determine categorical features.  

Taking a hint from our seasonal decomposition, it is natural to
select:  
- a feature capturing the 24-hour seasonality;  
- a feature capturing the 7-day seasonality; and  
- a feature capturing the (global) trend.  
Thus, with our data currently being from the same quarter of the same
year, we may think of taking:    
- the minute of day to capture the 24-hour seasonality;  
- the day of the week to capture the 7-day seasonality; and  
- any temporal feature of larger scale such as the week.  
However, random forests have no ability to extrapolate into the future
and therefore a third feature is unlikely to prove useful at all.  

### Setting the model's parameter
Random forests have many parameters and it can be quite daunting to
determine how to select them.  
- The number of trees over which the random forest takes averages to
  produce its outcome.  The larger this number, the more stable (hence
  reliable) the results.  Taking 100 trees produce consistent
  results.  
- The minimum number of samples in a leaf.  
- The maximum depth in the trees.  

Either the minimum number of samples in a leaf or the maximum depth in
the tree can be prescribed, but not both at the same time.  While
optimal values for either of these parameters can be found after trial
and error, it would be interesting to understand why these values
produce optimal results.  This is a question for further
investigation. 

### Training and validation sets
The following figure shows the splitting between training and
validation data for the location at rue du Fer à Moulin.

<p align="center">
<img src="moulin-training-validation-split.png" alt="Split between
training and validation data at rue du Fer à Moulin" width="1000"/>
</p>

### Random forests with maximum depth prescribed
The performance of the random forest is measured by the root mean
squared error (RMSE) and $R^2$ score.

Let's first look at the results with a maximum depth of 5.  

<p align="center"> <img
src="moulin-random-forest-forecast-max-depth-5.png" alt="Forecast of
seat availability at rue du Fer à Moulin using random forests with max
depth set to 5" width="1000"/> </p>

The predictions have the appearance of a discretized signal.  This is
actually not at all surprising.  Indeed, with a maximum depth of 5,
each tree can only output a maximum of 32 values, while the maximum
number of seats at the location is 63.  On the other hand, a random
forest takes averages of several decision trees (here 100), and thus
this averaging should produced more intermediate values.  This is not
what we observe in the graph.  The explanation is probably to be found
in the original signal (data) itself, which has quite a lot of
regularity.  This suggests that the trees of a random forest produce
nearly the same results.  In fact, experimenting with the number of
trees does not seem to make any changes, and using 5 trees (rather
than 100) produces the same result.

Furthermore, the random forest is quite effective during the fast
transition periods and not so good when availability is either low or high.  

Increasing now the maximum depth to 10 improves the predictions at low
and high availability.  

<p align="center"> <img
src="moulin-random-forest-forecast-max-depth-10.png" alt="Forecast of
seat availability at rue du Fer à Moulin using random forests with max depth
set to 10" width="1000"/> </p>

