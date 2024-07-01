---
title: "Hair and Makeup Wedding Vendors Database Design"
date: 2023-12-03
tags: [SQL, database, RDBMS, entities, ERD, functional dependencies, tables,data model]
header:
  image: "/images/hmua.jpg"
excerpt: "(SQL) Designed a database based on the hair and makeup vendors information in San Francisco Bay Area. This includes entities, functional dependencies, and Entity Relationship Diagram in 3NF."
mathjax: "true"
toc: true
toc_label : "Navigate":
---
By: Team Matrix <br>
Michelle Kae Celine Jo-anne Bantugon (Data Engineer) <br>
Miron Tkachuk (Data Engineer) <br>
Srinivas Kondeti (Product Manager) <br>
Zamambo Mkhize (Business-Side Stakeholder) <br>

Business Challenge built by [Professor Chase Kusterer](https://github.com/chase-kusterer)<br>
Hult International Business School<br>

### Functional Dependency
<img src="{{ site.url }}{{ site.baseurl }}/images/hmu_db_01.png" alt="">

### Entity Relationship Diagram (ERD)
<img src="{{ site.url }}{{ site.baseurl }}/images/hmu_db_02.png" alt="">

### SQL Queries
```
/*
Team 07 The Matrix (Vision Board 12)
Master’s degree on Business Analytics, Hult International Business School
BCH-7810: Business Challenge I
Professor Chase Kusterer
December 20, 2023
*/

/*
Hi Professor Chase. We utilized the same sql script from A2 Assignment. Please see at the end the wedding_cost_data query.
Thank you.
*/

-- --------------------------- --
-- ASSUMPTIONS / SPECIAL CASES --
-- --------------------------- --
/*
All departments are capable of accommodating various wedding capacities, except for the venue, for which we seek external 
information through outsourcing. In selecting a venue and dresses, we encountered limitations concerning different 
price levels, precluding us from finding suitable combinations for specific wedding sizes and budgetary considerations. 
While the database currently features a limited product selection for each vendor, we anticipate that they can also accommodate 
additional products available from other vendors. This assumption has been verified through the information provided on the 
website integrated into the database. This scenario is prevalent across the majority of departments.
*/

-- --------------------------- --
-- WEDDING SIZE / BUDGET LEVEL --
-- --------------------------- --

/*
Vendors were categorized into wedding size and budget level groups based on the price_ce attribute in the database. 
In specific instances where a department do not have some price_ce, we employ price ranges associated with each level 
to address null values in the relevant combinations.
In determining the total estimated cost, we assigned percentages to each element relative to its corresponding 
departmental cost, thereby reflecting its significance on our vision board.
*/
```
