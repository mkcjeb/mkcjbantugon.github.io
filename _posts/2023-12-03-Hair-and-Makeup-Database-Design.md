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
### Wedding Vendor Department
```
-- To access wedding database
USE Wedding_database;
```
### Dress and Attire Department
```
/*
DRESS AND ATTIRE DEPARTMENT
Groom and groomsmen
In light of the limited options available in the database for a particular shade of blue, we have sought alternative 
vendors that feature the desired color on their websites. This strategic approach aims to ensure a comprehensive array 
of choices while adhering to predefined budget constraints. Furthermore, recognizing the constraints in tie and lapel 
style choices within the existing database, we have carefully chosen vendors capable of accommodating the specific style 
preferences for both the groom and groomsmen, thereby ensuring a comprehensive and tailored selection process.
*/

/*
Table 1 (Groom and Groomsmen)
*/
DROP TABLE IF EXISTS attire_g;
CREATE TEMPORARY TABLE attire_g AS
SELECT 
    b.vendor_id, 
    b.vendor_depart,
    a.price_ce,
    color,
    tie,
	1 AS ws_small,
	1 AS ws_medium,
	1 AS ws_large
FROM products AS a
INNER JOIN vendors AS b 
ON b.vendor_id = a.vendor_id
INNER JOIN attire AS c
ON a.product_id = c.product_id
WHERE a.vendor_id LIKE '%att%'
    AND color IN ('dark blue', 'light blue', 'navy', 'black', 'green and blue') 
    AND tie IN ('bow tie', 'necktie') -- for groom and groomsmen
ORDER BY a.price_ce ASC;

/*
DRESS AND ATTIRE DEPARTMENT
Bride and bridesmaid
Wedding dress vendors have been carefully chosen based on stringent criteria pertaining to the quality of fabric and 
the specific silhouette of the dresses. The bride's dress encompasses exquisite fabrics such as lace and tulle, with 
available silhouettes including both mermaid and trumpet styles. Similarly, bridesmaid dress providers have been 
selected based on the dual criteria of sleeveless designs and satin fabric.
*/

-- Table 1 (Bride and Bridesmaid)
DROP TABLE IF EXISTS dress_b;
CREATE TEMPORARY TABLE dress_b AS
SELECT 
	b.vendor_id, 
    b.vendor_depart,
    a.price_ce,
    c.fabric,
    c.silhouette,
    c.neckline,
    c.sleeve,
	1 AS ws_small,
	1 AS ws_medium,
	1 AS ws_large
FROM products AS a
INNER JOIN vendors AS b
ON b.vendor_id = a.vendor_id
INNER JOIN dress AS c
ON a.product_id = c.product_id
WHERE a.vendor_id LIKE '%att%'
   AND c.fabric IN ('lace', 'tulle', 'satin')            	-- for the bride and bridesmaid
   AND c.silhouette IN ('mermaid', 'trumpet', 'a-line')  	-- for the bride and bridesmaid
   AND c.neckline LIKE '%v-neck%'        					-- for bridesmaid
   AND c.sleeve LIKE '%without sleeve%'  					-- for bridesmaid
ORDER BY a.price_ce;

/*
ATTIRE(GROOM) Table 2
*/
DROP TABLE IF EXISTS att_all_sizes;
CREATE TEMPORARY TABLE att_all_sizes AS
SELECT 
    'Small' AS wedding_size,
    CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
        WHEN a.price_ce = 2 THEN 'Affordable'
        WHEN a.price_ce = 3 THEN 'Moderate'
        WHEN a.price_ce = 4 THEN 'Luxury'
    END AS budget_level,
    1 AS attire_groom,
    ROUND(AVG(CASE WHEN c.color IN ('dark blue', 'light blue', 'navy', 'black', 'green and blue')
                    AND c.tie IN ('bow tie') THEN a.price_unit END), 2) AS attire_groom_price,
    1 AS attire_groomsmen,
    ROUND(AVG(CASE WHEN c.color IN ('dark blue', 'light blue', 'navy', 'black', 'green and blue')
                    AND c.tie IN ('necktie') THEN a.price_unit END), 2) AS attire_groomsmen_price
FROM products AS a
INNER JOIN attire_g AS c 
ON a.vendor_id = c.vendor_id
INNER JOIN vendors AS d 
ON a.vendor_id = d.vendor_id
WHERE ws_small = 1
GROUP BY 
	wedding_size, 
    budget_level, 
    attire_groom,
    attire_groomsmen;


/*
DRESS(BRIDE) Table 2
*/
DROP TABLE IF EXISTS dress_all_sizes;
CREATE TEMPORARY TABLE dress_all_sizes AS
SELECT 
    'Small' AS wedding_size,
    CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
        WHEN a.price_ce = 2 THEN 'Affordable'
        WHEN a.price_ce = 3 THEN 'Moderate'
        WHEN a.price_ce = 4 THEN 'Luxury'
    END AS budget_level,
    1 AS dress_bride,
        ROUND(AVG(CASE WHEN c.fabric IN ('lace', 'tulle')    -- bride only
						AND c.silhouette IN ('mermaid', 'trumpet')  
					   THEN a.price_unit END), 2) AS dress_bride_price,
	1 AS dress_bridesmaid,
        ROUND(AVG(CASE WHEN c.fabric LIKE '%satin%'           	-- bridesmaid only
						AND c.silhouette LIKE '%a-line%'  	
						AND c.neckline LIKE '%v-neck%'        					
						AND c.sleeve LIKE '%without sleeve%' 
					   THEN a.price_unit END), 2) AS dress_bridesmaid_price		
FROM products AS a
INNER JOIN dress_b AS c 
ON a.vendor_id = c.vendor_id
INNER JOIN vendors AS d 
ON a.vendor_id = d.vendor_id
WHERE ws_small = 1
GROUP BY 
	wedding_size, 
    budget_level, 
    dress_bride,
    dress_bridesmaid;
```



