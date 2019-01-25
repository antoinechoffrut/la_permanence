# Attendance at *La Permanence* coworking space #



[La Permanence](https://www.la-permanence.com "La Permanence coworking
space in Paris") offers two coworking spaces in Paris, in *rue du
Fer à Moulin* and *rue d'Alésia*.

This project monitors the number of available seats at the two
locations and performs some simple analysis.

There are three main components:  
  1. The data is collected with the script
     `la_permanence_scraping.py`.  
  2. The data is saved in `attendance.csv` (a misnomer, since it
records the number of available seats rather than the number of seats
used).  
  3. The Jupyter Notebook `la_permanence_EDA.ipynb` performs some
simple analysis of the data.  

The CSV file `attendance` contains three columns:  
  1. `timestamp`: date & time of collection of data, in the time
     standard **UTC** timezone.  The format is `YYYY-MM-DD-hh-mm-ss`
     with `YYYY=` year, `MM=` month, `DD=` day, `hh=` hour, `mm=`
     minute, `ss=` second   
  2. `Moulin`: number of *available* places at *rue du Fer à Moulin*  
  3. `Alésia`: number of *available* places at *rue d'Alésia*  
   
