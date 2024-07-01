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
### Access Wedding Vendor Database
```
-- To access wedding database
USE Wedding_database;
```
### Catering Department
```
/*
CATERING DEPARTMENT
Assumptions:
On average, caterers charge $500 to $7,500 for 50 guests, $1,000 to $15,000 for 100 guests, and $1,500 to $23,000 for 150 guests.
Cat_08 and Cat_50 are vendors offering at least Mediterranean cuisine, bar services, and cake cutting.
Cat_23, cat_29, cat_36, and cat_45 vendors are among the top 25% of vendors offering almost all cuisines. 
Additionally, they provide furnished tables and chairs, tableware, and bar services. 
Regarding dietary accommodations, they represent more than 50% of vendors offering all available options.
Catering department do not have price_ce of 1.
*/
-- Table 1
DROP TABLE IF EXISTS catering;
CREATE TEMPORARY TABLE catering AS
WITH rankedprices AS (
  SELECT
    a.vendor_id,
    a.price_ce,
    RANK() OVER (PARTITION BY a.vendor_id ORDER BY COUNT(*) DESC, MAX(a.price_ce) DESC) AS rank_pce
  FROM products AS a
  WHERE
    a.vendor_id LIKE '%cat%'
    AND a.vendor_id IN ('cat_08', 'cat_23', 'cat_29', 'cat_36', 'cat_45', 'cat_50')
  GROUP BY a.vendor_id, a.price_ce
)

SELECT
  b.vendor_id,
  b.vendor_depart,
  c.price_ce,
  1 AS ws_small,
  1 AS ws_medium,
  1 AS ws_large
FROM vendors AS b
INNER JOIN rankedprices AS c 
ON b.vendor_id = c.vendor_id AND c.rank_pce = 1
ORDER BY c.price_ce ASC;

/*
CATERING ALL_SIZES
*/
DROP TABLE IF EXISTS catering_all_size;
CREATE TEMPORARY TABLE catering_all_size AS
SELECT 	'Small' AS wedding_size,
	CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
		WHEN a.price_ce = 2 THEN 'Affordable'
		WHEN a.price_ce = 3 THEN 'Moderate'
		WHEN a.price_ce = 4 then 'Luxury'
	END AS budget_level,
		1 AS catering_cuisine,
		ROUND(AVG(CASE WHEN a.vendor_id IN ('cat_08', 'cat_23', 'cat_29', 'cat_36', 'cat_45', 'cat_50') THEN a.price_unit END), 2) 
          AS catering_cuisine_price
FROM products AS a
INNER JOIN catering AS b
ON a.vendor_id = b.vendor_id
WHERE ws_small = 1
GROUP BY 1,2,3;
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
### Flowers Department
```
/* For our vision board romantic and vintage styles are essential.
   However, we still can use flowers if they are summer/fall/spring season as they have appropriate color.
   Our vision board requires bouquet, flowers arrangements, boutounneries and flower petals.
*/
DROP TABLE IF EXISTS flower_products;
CREATE TEMPORARY TABLE flower_products as
SELECT vendor_id, p1.product_id, price_ce, product_name
FROM Products p1
JOIN Flower_Season_Style f1
ON f1.product_id = p1.product_id
WHERE 
(flower_style IN ('romantic', 'vintage') OR flower_season IN ('summer', 'fall', 'spring')) AND
product_name in ('bouquet', 'flowers arrangement', 'boutounneries', 'flower petals');

-- There are no limitations on flowers quantity on the websites of vendors, so we assume any vendor can serve any capacity
DROP TABLE IF EXISTS flowers;
CREATE TEMPORARY TABLE flowers as
SELECT v1.vendor_id, 
		vendor_depart,
        price_ce,
        1 as ws_small,
        1 as ws_medium,
        1 as ws_large
FROM Vendors v1
JOIN flower_products fp on fp.vendor_id = v1.vendor_id;

-- Flower Department for all wedding sizes
DROP TABLE IF EXISTS flowers_all_sizes;
CREATE TEMPORARY TABLE flowers_all_sizes as
select 	'small' as wedding_size,
 		case when f1.price_ce = 1 then 'Inexpensive'
			when f1.price_ce = 2 then 'Affordable'
			when f1.price_ce = 3 then 'Moderate'
            when f1.price_ce = 4 then 'Luxury'
            end as budget_level,
		1 as flowers_arrangement,
		ROUND(AVG(case when product_name like '%arr%' then price_unit end), 2) as arr_price,
        1 as flowers_bouquet,
        ROUND(AVG(case when product_name like '%bou%' then price_unit end), 2) as bou_price
from Products p1
INNER JOIN Flowers f1
ON f1.vendor_id = p1.vendor_id
WHERE ws_small = 1
group by 1,2,3,5;
```
### Invitation Department
```
/*
INVITATION DEPARTMENT
Invitation vendors are all online, and two vendors were selected based on the wedding theme, which is a 
minimalist design with a touch of greenery. The two vendors offer all the types of cards needed by the 
client, including the invitation, menu, place card, RSVP, table number, and wedding program. The client 
has a variety of options from these two vendors in different paper types for every category.
*/
-- Table 1
DROP TABLE IF EXISTS invitation;
CREATE TEMPORARY TABLE invitation AS
SELECT 
	d.vendor_id, 
    d.vendor_depart,
    a.price_ce,
	1 AS ws_small,
	1 AS ws_medium,
	1 AS ws_large
FROM products AS a
INNER JOIN inv_characteristics AS b
ON a.vendor_id = b.vendor_id
INNER JOIN inv_mailing AS c
ON b.mailing_id = c.mailing_id
INNER JOIN vendors AS d
ON a.vendor_id = b.vendor_id
WHERE d.vendor_id LIKE '%inv_%' AND d.vendor_name = 'coffee n cream press'
								OR (d.vendor_name = 'theknot' AND 
                                      ( d.vendor_website LIKE '%opulences-vera-wang%' OR 
									    d.vendor_website LIKE '%dashboard%' OR
                                        d.vendor_website LIKE '%timeless-frame%'
									   )
									)
ORDER BY a.price_ce ASC
;

/*
Invitation Final
*/
DROP TABLE IF EXISTS invitation_all_sizes;
CREATE TABLE invitation_all_sizes as 
SELECT 	'Small' AS wedding_size,
	CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
		WHEN a.price_ce = 2 THEN 'Affordable'
		WHEN a.price_ce = 3 THEN 'Moderate'
		WHEN a.price_ce = 4 then 'Luxury'
	END AS budget_level,
		1 AS invitation,
		ROUND(AVG(CASE WHEN d.vendor_id LIKE '%inv_%' AND d.vendor_name = 'coffee n cream press'
								OR (d.vendor_name = 'theknot' AND 
                                      ( d.vendor_website LIKE '%opulences-vera-wang%' OR 
									    d.vendor_website LIKE '%dashboard%' OR
                                        d.vendor_website LIKE '%timeless-frame%'
									  )
									)
			      THEN a.price_unit END), 2) AS invitation_price								
FROM products AS a
INNER JOIN invitation AS b 
ON a.vendor_id = b.vendor_id
INNER JOIN vendors AS d 
ON a.vendor_id = b.vendor_id
WHERE ws_small = 1
GROUP BY 1,2,3;
```
### Jewelry Department
```
/* Jewelry Department
   We choose only vendors with rings
   There are products for each price level provided by chosen vendors, for simplicity we assign average price_ce to each vendor here
*/
DROP TABLE IF EXISTS jewelry;
CREATE TEMPORARY TABLE jewelry AS
SELECT 
   v1.vendor_id,
   vendor_depart,
   ROUND(AVG(price_ce),0) as price_ce, 
   1 as ws_small,
   1 as ws_medium,
   1 as ws_large
FROM Products p1
JOIN Vendors v1
ON v1.vendor_id = p1.vendor_id
WHERE product_name LIKE '%ring%'
AND vendor_depart = 'jewelry'
GROUP BY 1,2,4,5,6 
;

-- Jewelry Department for all wedding sizes
DROP TABLE IF EXISTS jewelry_all_sizes;
CREATE TEMPORARY TABLE jewelry_all_sizes as
SELECT 	'All' AS wedding_size,
 		CASE WHEN f1.price_ce = 1 THEN 'Inexpensive'
			WHEN f1.price_ce = 2 THEN 'Affordable'
			WHEN f1.price_ce = 3 THEN 'Moderate'
            WHEN f1.price_ce = 4 THEN 'Luxury'
            END AS budget_level,
		1 AS jewelry,
		ROUND(AVG(price_unit), 2) as jewelry_price
FROM Products p1
INNER JOIN jewelry f1
ON f1.vendor_id = p1.vendor_id
group by 1,2,3;
```

### Hair and Makeup Department
```
/*
HAIR AND MAKEUP DEPARTMENT
To streamline vendor selection, proximity to the potential event venue is a key criterion.
Hair:
Our choice of hair providers is guided by the client's desired style. The bride opts for a 
'half-up' do with a braid, while the bridesmaids prefer elegant 'updo' chignons.
Makeup:
All makeup vendors provide both traditional and airbrush makeup styles, along with trial 
sessions for the client to choose her preferred look. Simple makeup options are available 
for brides, grooms, and children on both the bride and groom sides.
*/

-- Table 1
DROP TABLE IF EXISTS hmu;
CREATE TEMPORARY TABLE hmu AS
WITH rankedprices AS (
  SELECT
    a.vendor_id,
    a.price_ce,
    RANK() OVER (PARTITION BY a.vendor_id ORDER BY COUNT(*) DESC, MAX(a.price_ce) DESC) AS rank_pce
  FROM products AS a
  INNER JOIN vendors AS b
    ON a.vendor_id = b.vendor_id
  WHERE a.vendor_id LIKE '%hmu%'
    AND b.vendor_location IN ('berkeley', 'burlingame', 'san francisco', 'oakland', 'san jose')
  GROUP BY a.vendor_id, a.price_ce
)
SELECT
  b.vendor_id,
  b.vendor_depart,
  c.price_ce,
  1 AS ws_small,
  1 AS ws_medium,
  1 AS ws_large
FROM vendors AS b
INNER JOIN rankedprices AS c 
  ON b.vendor_id = c.vendor_id AND c.rank_pce = 1
ORDER BY c.price_ce ASC;

/*
HMU_ALL_SIZES Table 2
*/
DROP TABLE IF EXISTS hmu_all_sizes;
CREATE TEMPORARY TABLE hmu_all_sizes AS
SELECT 	'Small' AS wedding_size,
	CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
		WHEN a.price_ce = 2 THEN 'Affordable'
		WHEN a.price_ce = 3 THEN 'Moderate'
		WHEN a.price_ce = 4 then 'Luxury'
	END AS budget_level,
		1 AS hair_bride,
		ROUND(AVG(CASE WHEN a.product_name IN ('half up') AND c.vendor_location IN ('berkeley', 'burlingame', 'san francisco', 'oakland', 'san jose')
					   THEN a.price_unit END), 2) AS hair_bride_price,
		1 AS makeup_tr_bride,
        ROUND(AVG(CASE WHEN a.product_name IN ('traditional') AND c.vendor_location IN ('berkeley', 'burlingame', 'san francisco', 'oakland', 'san jose')
					   THEN a.price_unit END), 2) AS makeup_tr_bride_price,
		1 AS makeup_ab_bride,
        ROUND(AVG(CASE WHEN a.product_name IN ('airbrush') AND c.vendor_location IN ('berkeley', 'burlingame', 'san francisco', 'oakland', 'san jose')
					   THEN a.price_unit END), 2) AS makeup_ab_bride_price
					
FROM products AS a
INNER JOIN hmu AS b
ON a.vendor_id = b.vendor_id
INNER JOIN vendors AS c
ON a.vendor_id = c.vendor_id
WHERE ws_small = 1
GROUP BY 1,2,3,5;
```
### Music Department
```
/*
Music Department
We chose vendors based on their proximity to potential wedding venues.
*/
DROP TABLE IF EXISTS music;
CREATE TEMPORARY TABLE music AS
SELECT 
   vendor_id, 
   vendor_depart,
   price_ce, 
   1 as ws_small,
   1 as ws_medium,
   1 as ws_large
FROM Vendors
INNER JOIN Products USING(vendor_id)
WHERE 
   vendor_id LIKE '%dj%';

-- Music Department for all wedding sizes
DROP TABLE IF EXISTS music_all_sizes;
CREATE TEMPORARY TABLE music_all_sizes as
SELECT 
    'small' as wedding_size,
    CASE 
        WHEN price_ce = 1 THEN 'Inexpensive'
        WHEN price_ce = 2 THEN 'Affordable'
        WHEN price_ce = 3 THEN 'Moderate'
        WHEN price_ce = 4 THEN 'Luxury'
    END as budget_level,
    1 as music,
    -- Calculating the average popularity_id for products containing 'dj' in the name
    ROUND(AVG(CASE WHEN product_name LIKE '%dj%' THEN price_unit END), 2) as music_price
FROM 
    Products p1
GROUP BY 
    1, 2, 3
HAVING music_price is not null;
```
### Photo and Video Department
```
/*
Photo and Video Department
We chose vendors based on their proximity to potential wedding venues.
*/
DROP TABLE IF EXISTS photo_video;
CREATE TEMPORARY TABLE photo_video AS
SELECT 
   vendor_id,
   vendor_depart,
   price_ce, 
   1 as ws_small,
   1 as ws_medium,
   1 as ws_large
FROM Vendors
INNER JOIN Products USING(vendor_id)
WHERE 
   vendor_id LIKE '%vid%'
   AND (
      vendor_location LIKE '%san francisco%' 
       OR vendor_location LIKE '%san Jose%'
       OR vendor_location LIKE '%berkeley%'
       OR vendor_location LIKE '%san mateo%' 
       OR vendor_location LIKE '%dixon%'
       OR vendor_location LIKE '%pescadero%'
       OR vendor_location LIKE '%novato%'
       OR vendor_location LIKE '%walnut creek%'
       OR vendor_location LIKE '%menlo park%'
       OR vendor_location LIKE '%los gatos%'
       OR vendor_location LIKE '%sunol%'
       OR vendor_location LIKE '%calistoga%'
       OR vendor_location LIKE '%oakley%'
       OR vendor_location LIKE '%half moon bay%'
   );
   
-- Photo and Video Department for all wedding sizes
DROP TABLE IF EXISTS photo_all_sizes;
CREATE TEMPORARY TABLE photo_all_sizes as
SELECT 	'Small' AS wedding_size,
 		CASE WHEN f1.price_ce = 1 THEN 'Inexpensive'
			WHEN f1.price_ce = 2 THEN 'Affordable'
			WHEN f1.price_ce = 3 THEN 'Moderate'
            WHEN f1.price_ce = 4 THEN 'Luxury'
            END AS budget_level,
		1 AS photo_video,
		ROUND(AVG(price_unit), 2) as photo_price
FROM Products p1
INNER JOIN photo_video f1
ON f1.vendor_id = p1.vendor_id
group by 1,2,3;
```
### Rentals Department
```
/*
RENTALS DEPARTMENT
Vendors are filtered based on the type of rental products they can offer that our client needs for the wedding theme. 
This includes those who have the blue-colored goblets and candelabra. Though a maximum of three types is available in 
the database, these vendors offer all the types of rentals the client needs, as checked on their website. Their locations 
also differ from one another, providing more choices for the client.
Most rental companies have a minimum order amount for delivery. 
ren_10 glassware (san jose) no information online, 
ren_13 linen, dinnerware (brisbane) minimum order for delivery is $1,000.00 in rentals plus delivery charges, 
ren_16 linen, chairs (oakland) minimum order is $250 plus delivery charges,
ren_12 glasswares tables (hayward) minimum delivery fee is $150,
ren_21 glassware, tables, and chairs (san jose) the total price of an order must meet or exceed a minimum cost threshold, 
ren_10 dinnerware, chairs (san jose) no information online.
*/
-- Table 1
DROP TABLE IF EXISTS rentals;
CREATE TEMPORARY TABLE rentals AS
WITH rankedprices AS (
  SELECT
    a.vendor_id,
    a.price_ce,
    RANK() OVER (PARTITION BY a.vendor_id ORDER BY COUNT(*) DESC, MAX(a.price_ce) DESC) AS rank_pce
  FROM products AS a
  WHERE a.vendor_id LIKE '%ren%'
    AND a.vendor_id IN ('ren_10', 'ren_12', 'ren_13', 'ren_16', 'ren_17', 'ren_21')
  GROUP BY a.vendor_id, a.price_ce
)
SELECT
  b.vendor_id,
  b.vendor_depart,
  c.price_ce,
  1 AS ws_small,
  1 AS ws_medium,
  1 AS ws_large
FROM vendors AS b
INNER JOIN rankedprices AS c 
ON b.vendor_id = c.vendor_id AND c.rank_pce = 1
ORDER BY c.price_ce ASC;

/*
RENTAL_ALL_SIZES Table 2
*/
DROP TABLE IF EXISTS rentals_all_sizes;
CREATE TEMPORARY TABLE rentals_all_sizes AS
SELECT 	'Small' AS wedding_size,
	CASE 
        WHEN a.price_ce = 1 THEN 'Inexpensive'
		WHEN a.price_ce = 2 THEN 'Affordable'
		WHEN a.price_ce = 3 THEN 'Moderate'
		WHEN a.price_ce = 4 then 'Luxury'
	END AS budget_level,
		1 AS rentals_table,
		COALESCE(ROUND(AVG(CASE WHEN a.product_name LIKE '%table%' THEN a.price_unit END), 2),
        CASE 
        WHEN a.price_ce = 1 THEN 0.85
		WHEN a.price_ce = 2 THEN 3
		WHEN a.price_ce = 3 THEN 11.5
		WHEN a.price_ce = 4 then 300
		END
        ) AS rentals_tbl_price,
        1 AS rentals_glassware,
        COALESCE(ROUND(AVG(CASE WHEN a.product_name LIKE '%glass%' THEN a.price_unit END), 2),
        CASE 
        WHEN a.price_ce = 1 THEN 0.85
		WHEN a.price_ce = 2 THEN 3
		WHEN a.price_ce = 3 THEN 11.5
		WHEN a.price_ce = 4 then 300
		END
        ) AS rentals_gw_price
FROM products AS a
INNER JOIN rentals AS b
ON a.vendor_id = b.vendor_id
WHERE ws_small = 1
GROUP BY 1,2,3,5, a.price_ce;
```
### Venue Department
```
/* 
   Venues Department
   Sizes of venues are taken from outsource resources (websites) and assigned based on that
   For our vision board style we choose garden venues or ballrooms with both indoor and outdoor areas.
   We lack small and medium size venues in our database
*/
DROP TABLE IF EXISTS venues;
CREATE TEMPORARY TABLE venues as
WITH t1 AS (
SELECT DISTINCT vendor_id, price_ce
FROM Products p1
JOIN ven_amenities v1
ON p1.product_id = v1.product_id
JOIN ven_settings v2
ON p1.product_id = v2.product_id
WHERE 
(ven_type = 'garden' or ven_type = 'ballroom')
OR price_ce = 1)
SELECT v1.vendor_id, 
		vendor_depart,
        price_ce,
        CASE WHEN v1.vendor_id in ('ven_29', 'ven_31', 'ven_37') THEN 1
			 ELSE 0 
             END as ws_small,
		CASE WHEN v1.vendor_id in ('ven_05','ven_07','ven_23','ven_29','ven_31','ven_37','ven_47','ven_50') THEN 1
			 ELSE 0
             END as ws_medium,
		CASE WHEN v1.vendor_id in ('ven_05','ven_07','ven_23','ven_29','ven_31','ven_37','ven_47','ven_50') THEN 0
			 ELSE 1
             END as ws_large
FROM Vendors v1
JOIN t1 on t1.vendor_id = v1.vendor_id;


-- Venue Department for small wedding size
DROP TABLE IF EXISTS venues_small;
CREATE TEMPORARY TABLE venues_small as
SELECT 	'Small' AS wedding_size,
 		CASE WHEN f1.price_ce = 1 THEN 'Inexpensive'
			WHEN f1.price_ce = 2 THEN 'Affordable'
			WHEN f1.price_ce = 3 THEN 'Moderate'
            WHEN f1.price_ce = 4 THEN 'Luxury'
            END AS budget_level,
		1 AS venue,
		ROUND(AVG(price_unit),2) as venue_price
FROM Products p1
INNER JOIN venues f1
ON f1.vendor_id = p1.vendor_id
WHERE ws_small = 1
group by 1,2,3;

-- Venue Department for medium wedding size
DROP TABLE IF EXISTS venues_medium;
CREATE TEMPORARY TABLE venues_medium as
SELECT 	'Medium' AS wedding_size,
 		CASE WHEN f1.price_ce = 1 THEN 'Inexpensive'
			WHEN f1.price_ce = 2 THEN 'Affordable'
			WHEN f1.price_ce = 3 THEN 'Moderate'
            WHEN f1.price_ce = 4 THEN 'Luxury'
            END AS budget_level,
		1 AS venue,
		ROUND(AVG(price_unit),2) as venue_price
FROM Products p1
INNER JOIN venues f1
ON f1.vendor_id = p1.vendor_id
WHERE ws_medium = 1
group by 1,2,3;

-- Venue Department for large wedding size
DROP TABLE IF EXISTS venues_large;
CREATE TEMPORARY TABLE venues_large as
SELECT 	'Large' AS wedding_size,
 		CASE WHEN f1.price_ce = 1 THEN 'Inexpensive'
			WHEN f1.price_ce = 2 THEN 'Affordable'
			WHEN f1.price_ce = 3 THEN 'Moderate'
            WHEN f1.price_ce = 4 THEN 'Luxury'
            END AS budget_level,
		1 AS venue,
		ROUND(AVG(price_unit), 2) as venue_price
FROM Products p1
INNER JOIN venues f1
ON f1.vendor_id = p1.vendor_id
WHERE ws_large = 1
group by 1,2,3;
```
### Budget Level and Wedding Size Different Combinations
```
-- Table 2 Template for Budget Level and Wedding Size Different Combinations
DROP TABLE IF EXISTS TEMPLATE;
CREATE TEMPORARY TABLE TEMPLATE(
wedding_size varchar(255), 
budget_level varchar(255));

INSERT INTO TEMPLATE VALUES
('Small', 'Inexpensive'),
('Small', 'Affordable'),
('Small', 'Moderate'),
('Small', 'Luxury'),
('Medium', 'Inexpensive'),
('Medium', 'Affordable'),
('Medium', 'Moderate'),
('Medium', 'Luxury'),
('Large', 'Inexpensive'),
('Large', 'Affordable'),
('Large', 'Moderate'),
('Large', 'Luxury');

/* If there is no price for some combination then we use ranges of price_ce provided by each department and take max value for ranges 1,2,3
   And avg value for price_ce = 4 (Luxury)
*/
DROP TABLE IF EXISTS SECOND_FINAL_TABLE_PREPARATION;
CREATE TEMPORARY TABLE SECOND_FINAL_TABLE_PREPARATION AS
SELECT  t1.wedding_size,
		t1.budget_level,
        'romantic/vintage' as wedding_theme,
        flowers_arrangement,
        arr_price,
        flowers_bouquet,
        bou_price,
        COALESCE(v1.venue, v2.venue, v3.venue, 1) as venue,
        COALESCE(v1.venue_price, v2.venue_price, v3.venue_price,
				CASE
                      WHEN t1.budget_level = 'Inexpensive' THEN 4000
					  WHEN t1.budget_level = 'Affordable' THEN 6000
					  WHEN t1.budget_level = 'Moderate' THEN 10000
					  WHEN t1.budget_level = 'Luxury' then 30000
				END
				) as venue_price,
		CASE
                      WHEN t1.wedding_size = 'Small' THEN 50
					  WHEN t1.wedding_size = 'Medium' THEN 100
					  WHEN t1.wedding_size = 'Large' THEN 300
		END as invitation,
        CASE
                      WHEN t1.wedding_size = 'Small' THEN i1.invitation_price * 0.5
					  WHEN t1.wedding_size = 'Medium' THEN i1.invitation_price * 1
					  WHEN t1.wedding_size = 'Large' THEN i1.invitation_price * 3
		END as invitation_price,
		1 as catering_cuisine,
        coalesce(catering_cuisine_price, 50) as catering_cuisine_price,
        rentals_table,
        rentals_tbl_price,
        rentals_glassware,
        rentals_gw_price,
        hair_bride,
        hair_bride_price,
        makeup_tr_bride,
        makeup_tr_bride_price,
        makeup_ab_bride,
        makeup_ab_bride_price,
        music,
        music_price,
        photo_video,
        photo_price,
        attire_groom,
        attire_groom_price,
        attire_groomsmen,
        attire_groomsmen_price,
        dress_bride,
        COALESCE(dress_bride_price,
				 CASE
                      WHEN t1.budget_level = 'Inexpensive' THEN 375
					  WHEN t1.budget_level = 'Affordable' THEN 469
					  WHEN t1.budget_level = 'Moderate' THEN 522.5
					  WHEN t1.budget_level = 'Luxury' then 1500
				 END) as dress_bride_price,
        dress_bridesmaid,
        COALESCE(dress_bridesmaid_price,
				 CASE
                      WHEN t1.budget_level = 'Inexpensive' THEN 375
					  WHEN t1.budget_level = 'Affordable' THEN 469
					  WHEN t1.budget_level = 'Moderate' THEN 522.5
					  WHEN t1.budget_level = 'Luxury' then 1500
				 END) as dress_bridesmaid_price,
		1 as jewelry,
        COALESCE(jewelry_price,
				CASE
                      WHEN t1.budget_level = 'Inexpensive' THEN 950
					  WHEN t1.budget_level = 'Affordable' THEN 1895
					  WHEN t1.budget_level = 'Moderate' THEN 3100
					  WHEN t1.budget_level = 'Luxury' then 5000
				 END) as jewelry_price
FROM TEMPLATE t1
LEFT JOIN flowers_all_sizes f1
ON f1.budget_level = t1.budget_level
LEFT JOIN venues_small v1
ON v1.wedding_size = t1.wedding_size AND v1.budget_level = t1.budget_level
LEFT JOIN venues_medium v2
ON v2.wedding_size = t1.wedding_size AND v2.budget_level = t1.budget_level
LEFT JOIN venues_large v3
ON v3.wedding_size = t1.wedding_size AND v3.budget_level = t1.budget_level
LEFT JOIN invitation_all_sizes i1 
ON i1.budget_level = t1.budget_level
LEFT JOIN catering_all_size c1
ON c1.budget_level = t1.budget_level
LEFT JOIN rentals_all_sizes r1
ON r1.budget_level = t1.budget_level
LEFT JOIN hmu_all_sizes h1
ON h1.budget_level = t1.budget_level
LEFT JOIN music_all_sizes m1
ON m1.budget_level = t1.budget_level
LEFT JOIN photo_all_sizes p1
ON p1.budget_level = t1.budget_level
LEFT JOIN dress_all_sizes d1
ON d1.budget_level = t1.budget_level
LEFT JOIN att_all_sizes a1
ON a1.budget_level = t1.budget_level
LEFT JOIN jewelry_all_sizes j1
ON j1.budget_level = t1.budget_level
;
```
### Assigned Weights based on importance
```
/* 
   We assigned weights based on importance for the style and on amount of products will be used from that department
   DROP TABLE IF EXISTS SECOND_FINAL_TABLE;
   CREATE TEMPORARY TABLE SECOND_FINAL_TABLE AS
*/
SELECT *,
		5 * flowers_arrangement * arr_price + 1*flowers_bouquet * bou_price + 
        1.5 * venue * venue_price  +
        invitation_price +
        2 * catering_cuisine_price * catering_cuisine +
        rentals_table * rentals_tbl_price +
        rentals_glassware * rentals_gw_price +
        3 * hair_bride * hair_bride_price +
        makeup_tr_bride * makeup_tr_bride_price +
        makeup_ab_bride * makeup_ab_bride_price +
        music * music_price +
        photo_video * photo_price +
        attire_groom * attire_groom_price +
        attire_groomsmen * attire_groomsmen_price +
        dress_bride * dress_bride_price +
        dress_bridesmaid_price * dress_bridesmaid +
        jewelry * jewelry_price
        as wedding_price
FROM SECOND_FINAL_TABLE_PREPARATION;
```
### All information from each department
```
-- Selecting all information from each department with different wedding size and budget level
DROP TABLE IF EXISTS FIRST_FINAL_TABLE;
CREATE TEMPORARY TABLE FIRST_FINAL_TABLE AS
SELECT * FROM photo_video
UNION ALL
SELECT * FROM music
UNION ALL
SELECT * FROM hmu
UNION ALL
SELECT * FROM  rentals
UNION ALL
SELECT * FROM  catering
UNION ALL
SELECT * FROM  invitation
UNION ALL
SELECT * FROM flowers
UNION ALL
SELECT * FROM Venues
UNION ALL
SELECT vendor_id, vendor_depart, price_ce, ws_small, ws_medium, ws_large FROM attire_g
UNION ALL
SELECT vendor_id, vendor_depart, price_ce, ws_small, ws_medium, ws_large FROM dress_b
UNION ALL
SELECT * FROM jewelry;
```
### Assigned Wedding Size (Medium) & Budget Level (Moderate) 
```
/* 
ALL VENDORS ARE CHOSEN FROM FIRST_FINAL_TABLE AND HAVE PRICE_CE = 3 (EXCEPT PHOTO AND JEWELRY) AND WS_LARGE = 1
final products for jewelry
*/
DROP TABLE IF EXISTS jewelry_products;
CREATE TEMPORARY TABLE jewelry_products AS
SELECT  vendor_depart as department,
		vendor_name,
        2450 as vendor_budget,
        'Wedding Ring' as product_name,
        price_unit as price_per_item,
        2 as quantity,
        2*price_unit as subtotal
FROM Vendors v1
INNER JOIN Products p1
ON v1.vendor_id = p1.vendor_id
WHERE product_name like '%volary ring%'
AND v1.vendor_id = 'jwl_04';

-- final products for music
DROP TABLE IF EXISTS music_products;
CREATE TEMPORARY TABLE music_products AS
SELECT  vendor_depart as department,
		vendor_name,
        2080 as vendor_budget,
        'Music' as product_name,
        price_unit as price_per_item,
        1 as quantity,
        1*price_unit as subtotal
FROM Vendors v1
INNER JOIN Products p1
ON v1.vendor_id = p1.vendor_id
WHERE product_name like '%dj%'
AND v1.vendor_id = 'dj_12';

-- final products for photo and video
DROP TABLE IF EXISTS photo_products;
CREATE TEMPORARY TABLE photo_products AS
SELECT  vendor_depart as department,
		vendor_name,
        3940 as vendor_budget,
        'Wedding Photoshoot' as product_name,
        price_unit as price_per_item,
        1 as quantity,
        1*price_unit as subtotal
FROM Vendors v1
INNER JOIN Products p1
ON v1.vendor_id = p1.vendor_id
WHERE v1.vendor_id = 'vid_08';

-- final products for venue
DROP TABLE IF EXISTS venue_products;
CREATE TEMPORARY TABLE venue_products AS
SELECT  vendor_depart as department,
		vendor_name,
        17800 as vendor_budget,
        'Wedding Venue' as product_name,
        price_unit as price_per_item,
        1 as quantity,
        1*price_unit as subtotal
FROM Vendors v1
INNER JOIN Products p1
ON v1.vendor_id = p1.vendor_id
WHERE v1.vendor_id = 'ven_46';

-- final table which would combine everything
DROP TABLE IF EXISTS wedding_cost_data;
CREATE TABLE wedding_cost_data
(category varchar(255), 
vendor_name varchar(255), 
budget_level int, 
item_name varchar(255), 
price_per_item float,
quantity int,
subtotal float);
```
### Outsourcing
```
-- Flowers outsource data
INSERT INTO wedding_cost_data VALUES
('flowers', 'Expressions Floral', 980, 'Wonderful White Bouquet of Flowers', 65, 1, 65*1),
('flowers', 'Expressions Floral', 980, 'Classic White Bridal Bouquet', 65, 1, 65*1),
('flowers', 'Expressions Floral', 980, 'Whimsical Wispies Bouquet', 160, 1, 160);

-- Catering outsource data
INSERT INTO wedding_cost_data VALUES
('catering', 'Fogcutter', 91.75 * 137, 'Plated Serving', 110, 137*0.83, 137*110*0.83),
('catering', 'Fogcutter', 91.75 * 137, 'Panna Cotta shooters', 3, 137*0.83, 3*137*0.83),
('catering', 'Fogcutter', 91.75 * 137, 'Three Tier Cake', 350, 1, 350*1);

-- Invitations outsource data
INSERT INTO wedding_cost_data VALUES
('invitation', 'Paperculture', 920, 'RSVP', 3.93, 135, 135*3.93),
('invitation', 'Theknot', 920, 'Invitation', 0.89, 135, 0.89*135),
('invitation', 'Theknot', 920, 'Envelope liner', 3.52, 135, 135*3.52),
('invitation', 'Theknot', 920, 'Wedding Program', 0.5, 135, 135*0.5),
('invitation', 'Paperculture', 920, 'Menu', 1.18, 135, 1.18*135);

-- Dress and Attire outsource data
INSERT INTO wedding_cost_data VALUES
('dress and attire', 'Blacktux', 486 + 486 + 450 + 522.5, "Groom's suit and bowtie", 159, 1, 159),
('dress and attire', 'Blacktux', 486 + 486 + 450 + 522.5, "Groomsmen suit and tie", 159, 8, 159*8),
('dress and attire', 'Stacees', 486 + 486 + 450 + 522.5, "Bridal Gown", 417, 1, 417),
('dress and attire', 'Misaac', 486 + 486 + 450 + 522.5, "Bridemaid's Dresses", 99, 8, 792.00);

-- Hair and Makeup outsource data
INSERT INTO wedding_cost_data VALUES
('hair and makeup', 'Shineforth Salon', 3*177.5 + 198 + 228, "Bridal Party Hair & Makeup (Airbrush)", 275, 9, 275*9),
('hair and makeup', 'The Pretty Committee', 3*177.5 + 198 + 228, "Groom Party Hair & Makeup (Simple)", 63, 9, 63*9);

-- Inserting all the data into the final table
INSERT INTO wedding_cost_data
SELECT * FROM jewelry_products;

INSERT INTO wedding_cost_data
SELECT * FROM music_products;

INSERT INTO wedding_cost_data
SELECT * FROM photo_products;

INSERT INTO wedding_cost_data
SELECT * FROM venue_products;

-- New rental vendor from outsource resource (requirement of the client)
/* INSERT QUERY NO: 1 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'cross back farm chair', 4.00, 114, 456.00
);

/* INSERT QUERY NO: 2 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'clear chiavari chairs', 5.00, 114, 570.00
);

/* INSERT QUERY NO: 3 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'beige loveseat', 40.00, 1, 40.00
);

/* INSERT QUERY NO: 4 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'farmhouse tables rental 8 pax', 20.00, 17, 340.00
);

/* INSERT QUERY NO: 5 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'natural sweetheart table', 30.00, 1, 30.00
);

/* INSERT QUERY NO: 6 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white dessert table', 50.00, 4, 200.00
);

/* INSERT QUERY NO: 7 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white vintage table', 20.00, 1, 20.00
);

/* INSERT QUERY NO: 8 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'wooden stumps or wood slices for cake', 2.50, 1, 2.50
);

/* INSERT QUERY NO: 9 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'lavender and eucalyptus garland 6 ft', 6.00, 18, 108.00
);

/* INSERT QUERY NO: 10 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'table garland', 10.00, 18, 180.00
);

/* INSERT QUERY NO: 11 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'table numbers', 2.50, 17, 42.50
);

/* INSERT QUERY NO: 12 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'dusty blue table runner 10 ft', 4.00, 18, 72.00
);

/* INSERT QUERY NO: 13 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'dusty blue drape 10 ft', 4.00, 18, 72.00
);

/* INSERT QUERY NO: 14 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'dusty blue napkins linen', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 15 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'boxwood greenery backdrop', 30.00, 1, 30.00
);

/* INSERT QUERY NO: 16 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'copper background stand or arch', 25.00, 1, 25.00
);

/* INSERT QUERY NO: 17 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white chiffon backdrop drape', 9.00, 1, 9.00
);

/* INSERT QUERY NO: 18 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, '6 ft white peony garland', 10.00, 1, 10.00
);

/* INSERT QUERY NO: 19 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'walnut wood welcome sign', 75.00, 1, 75.00
);

/* INSERT QUERY NO: 20 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'gold flatware set 3 piece', 1.00, 114, 114.00
);

/* INSERT QUERY NO: 21 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'gold beaded glass chargers', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 22 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white dinner plate', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 23 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white dessert plates', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 24 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'white salad plates', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 25 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'champagne flutes', 0.50, 114, 57.00
);

/* INSERT QUERY NO: 26 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'dark blue goblet', 1.00, 114, 114.00
);

/* INSERT QUERY NO: 27 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'light blue goblet', 1.00, 114, 114.00
);

/* INSERT QUERY NO: 28 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'gold tea light candle holders', 0.50, 26, 13.00
);

/* INSERT QUERY NO: 29 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'gold candle holders', 0.50, 60, 30.00
);

/* INSERT QUERY NO: 30 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'dusty blue 10 inches premium wax taper candles', 1.00, 60, 60.00
);

/* INSERT QUERY NO: 31 */
INSERT INTO wedding_cost_data(category, vendor_name, budget_level, item_name, price_per_item, quantity, subtotal)
VALUES
(
'rentals', 'eventlyst', 3100, 'wood and antiqued brass 2 tier serving dessert stand', 1.10, 8, 8.80
);
```

### Final table
```
-- Final table
SELECT *
FROM wedding_cost_data
limit 30;
```
### Table Output
<img src="{{ site.url }}{{ site.baseurl }}/images/wedding_table.png" alt="">


