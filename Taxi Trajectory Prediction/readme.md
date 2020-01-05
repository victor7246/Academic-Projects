### Project description

In this project, we predict current travel time between two location based on the historical data. Further we identify shortest route between two points using dynamic shortest path algorithm.

### Dataset

We have used dataset provided in Kaggle ECML/PKDD 15: Taxi Trajectory Prediction (I) competition (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data).

Dataset has a complete year (from 01/07/2013 to 30/06/2014) of the trajectories for all the 442 taxis running in the city of Porto, Portugal.

Each data sample corresponds to one completed trip. It contains a total of
9 (nine) features, described as follows:

* TRIP_ID: (String) It contains an unique identifier for each trip;
* CALL_TYPE: (char) It identifies the way used to demand this service. It may contain one of three possible values:
‘A’ if this trip was dispatched from the central;
‘B’ if this trip was demanded directly to a taxi driver on a specific stand;
‘C’ otherwise (i.e. a trip demanded on a random street).
* ORIGIN_CALL: (integer) It contains an unique identifier for each phone number which was used to demand, at least, one service. It identifies the trip’s customer if CALL_TYPE=’A’. Otherwise, it assumes a NULL value;
* ORIGIN_STAND: (integer): It contains an unique identifier for the taxi stand. It identifies the starting point of the trip if CALL_TYPE=’B’. Otherwise, it assumes a NULL value;
* TAXI_ID: (integer): It contains an unique identifier for the taxi driver that performed each trip;
* TIMESTAMP: (integer) Unix Timestamp (in seconds). It identifies the trip’s start; 
* DAYTYPE: (char) It identifies the daytype of the trip’s start. It assumes one of three possible values:
‘B’ if this trip started on a holiday or any other special day (i.e. extending holidays, floating holidays, etc.);
‘C’ if the trip started on a day before a type-B day;
‘A’ otherwise (i.e. a normal day, workday or weekend).
* MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete and TRUE whenever one (or more) locations are missing
* POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 format) mapped as a string. The beginning and the end of the string are identified with brackets (i.e. [ and ], respectively). Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. This list contains one pair of coordinates for each 15 seconds of trip. The last list item corresponds to the trip’s destination while the first one represents its start.

### Method overview

* We represent the whole city in terms of grids, where each pair of location is represented by the centre of the corresponding grids.
* The challenge is to predict time taken by taxi to travel from one grid to another. Using this, we can construct dynamic graph of all the roads in the whole city.
* For modelling, we use tree based methods - Randomforest, GradientBoosting
* Finally, we use shortest path algorithm to calculate the shortest path between two points at any given time of a given day.
