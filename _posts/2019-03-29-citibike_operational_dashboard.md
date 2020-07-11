---
title: "Citi Bike: Operational Dashboard"
date: 2019-03-29
tags: [SQL, google data studio, google big query, dashboards, data visualization]
header:
  image: "/images/citibike.jpg"
excerpt: "This Google Data Studio dashboard aims to satisfy CitiBike operational needs using live public data from Google Big Query."
mathjax: "true"
---

## Introduction

With the increase of traffic and full public transport options, bike sharing systems have become more and more common in the past 10 years. In most major cities we can now see visitors touristing the city on two wheels, or locals speeding by on their daily commutes. 

With the popularity of these systems, businesses such as Citi Bike in New York City have operational challenges such as capacity at stations and redistribution of bikes based on demand.

This dashboard aims to provide answers on any specific station's health or the overall system's health. The data is sourced from Big Query public data: new_york_citibike.citibike_trips .

Station Health
1. How many stations are at capacity, empty, or out of service?
2. What is the fill rate(bikes available/capacity) for each station?
3. What is the most popular station to start rides for all time?
4. What is the most popular station to end rides for all time?
5. What are the top 3 most popular trips (start and end station combination) for all time?
6. Which hours of the day does usage peak on weekdays?
7. Which hours of day does usage peak on weekends?

System Health
1. How many trips are there per day?
2. What is the average trip duration?
3. What was the shortest trip?
4. What was the longest trip?
5. How many total hours of usage does each bike have?


## Dashboard

The dashboard can be found [here](https://datastudio.google.com/s/oEq2kNO82HQ). Please note there are two pages, one for station and the other for system.


## SQL queries
In order to answer the above questions, I have connected Data Studio to Big Query, using SQL to extract the needed information.

### Station Health

* How many stations are at capacity, empty, or out of service?

```sql

SELECT (CASE WHEN num_docks_available = 0 THEN 'At Capacity'
             WHEN num_bikes_available = 0 THEN 'Empty'
             WHEN is_installed = false OR is_renting = false OR is_returning = false THEN 'Out of Service' 
             ELSE 'functioning' END) AS station_status
FROM `bigquery-public-data.new_york_citibike.citibike_stations`

```

* What is the fill rate(bikes available/capacity) for each station?

```sql

SELECT ROUND(num_bikes_available / capacity,2) as fill_rate, name
FROM `bigquery-public-data.new_york_citibike.citibike_stations`

```

* What is the most popular station to start rides for all time?

```sql

SELECT count(start_station_name) as num_rides, start_station_name
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
GROUP BY start_station_name
HAVING start_station_name NOT LIKE ''
ORDER BY num_rides desc
LIMIT 1;

```

* What is the most popular station to end rides for all time?

```sql

SELECT count(end_station_name) as num_rides, end_station_name
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
GROUP BY end_station_name
HAVING end_station_name NOT LIKE ''
ORDER BY num_rides desc
LIMIT 1;

```

* What are the top 3 most popular trips (start and end station combination) for all time?

```sql

SELECT start_station_name, end_station_name, COUNT(*) AS count
FROM `bigquery-public-data.new_york.citibike_trips`
GROUP BY start_station_name, end_station_name
ORDER BY count DESC
LIMIT 3;

```

* Which hours of the day does usage peak on weekdays?

```sql

SELECT COUNT(hour) as num_trips, hour
FROM(SELECT  EXTRACT(DAYOFWEEK FROM starttime) as day, EXTRACT(HOUR FROM starttime) as hour,*
     FROM `bigquery-public-data.new_york_citibike.citibike_trips`) as subquery
WHERE day IN (2,3,4,5,6)
GROUP BY hour
ORDER BY num_trips desc;

```

* Which hours of day does usage peak on weekends? 

```sql

SELECT COUNT(hour) as num_trips, hour
FROM(SELECT  EXTRACT(DAYOFWEEK FROM starttime) as day, EXTRACT(HOUR FROM starttime) as hour,*
     FROM `bigquery-public-data.new_york_citibike.citibike_trips`) as subquery
WHERE day NOT IN (2,3,4,5,6)
GROUP BY hour
ORDER BY num_trips desc;

```

### System Health

* How many trips are there per day?

```sql

SELECT COUNT(*) as num_trips, day
FROM(SELECT  EXTRACT(DATE FROM starttime) as day, *
     FROM `bigquery-public-data.new_york_citibike.citibike_trips`) as subquery
WHERE day IS NOT NULL
GROUP BY day
ORDER BY DAY ASC;

```

* What is the average trip duration?

```sql

SELECT ROUND(AVG(tripduration/60),2) as avg_duration_min
FROM `bigquery-public-data.new_york_citibike.citibike_trips`;

```

* What was the shortest trip?

```sql

SELECT ROUND(MIN(tripduration/60),2) as min_duration
FROM `bigquery-public-data.new_york_citibike.citibike_trips`;

```
* What was the longest trip?

```sql

SELECT ROUND(MAX(tripduration/60/60/24/30),1) as max_duration
FROM `bigquery-public-data.new_york_citibike.citibike_trips`;

```

* How many total hours of usage does each bike have?

```sql

SELECT bikeid, ROUND(SUM(tripduration/60/60),2) as hrs_usage
FROM `bigquery-public-data.new_york_citibike.citibike_trips`
group by bikeid
HAVING bikeid IS NOT NULL
order by hrs_usage desc;

```
