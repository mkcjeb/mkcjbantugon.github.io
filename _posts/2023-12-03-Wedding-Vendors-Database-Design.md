---
title: "Wedding Vendors Database Design"
date: 2023-12-03
tags: [SQL, database, RDBMS, entities, ERD, functional dependencies, tables,data model]
header:
  image: "/images/hmua.jpg"
excerpt: "(SQL) Designed a database based on the hair and makeup vendors information in San Francisco Bay Area. This includes entities, functional dependencies, and Entity Relationship Diagram in 3NF."
mathjax: "true"
toc: true
toc_label : "Navigate":
---
By: Team Matrix (Hair & Makeup Department) <br>
Michelle Kae Celine Jo-anne Bantugon (Data Engineer) <br>
Miron Tkachuk (Data Engineer) <br>
Srinivas Kondeti (Product Manager) <br>
Zamambo Mkhize (Business-Side Stakeholder) <br>

Business Challenge built by [Professor Chase Kusterer](https://github.com/chase-kusterer)<br>
Hult International Business School<br>

### I. Functional Dependency
<img src="{{ site.url }}{{ site.baseurl }}/images/hmu_db_01.png" alt="">

### II. Entity Relationship Diagram (ERD)
<img src="{{ site.url }}{{ site.baseurl }}/images/hmu_db_02.png" alt="">

### III. Integrating All Wedding Vendors (Cohort Project MBAN 2023)

### Dropping & Creating the Database
```
DROP DATABASE IF EXISTS Wedding_database;
CREATE DATABASE IF NOT EXISTS Wedding_database;


#ATTENTION: MOST OF NULL VALUES ARE NOT NULLS BUT EMPTY STRINGS IN OUR DATA
#UNFORTUNATELY!
# WHERE (COLUMN IS NULL OR COLUMN = '')
USE Wedding_database;
```

### Global Tables (Vendors & Products)
```
#creation of 2 global tables
DROP TABLE IF EXISTS Vendors;
CREATE TABLE Vendors (
	vendor_id VARCHAR(255) NOT NULL,
    vendor_depart VARCHAR(255) NOT NULL,
    vendor_name VARCHAR(255) NOT NULL,
    vendor_location VARCHAR(255),
    vendor_sustainable BOOL,
    vendor_rating SMALLINT UNSIGNED,
    vendor_contact VARCHAR(255),
    vendor_website VARCHAR(255),
    PRIMARY KEY (vendor_id)
);


DROP TABLE IF EXISTS Products;
CREATE TABLE Products (
	product_id VARCHAR(255) NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    vendor_id VARCHAR(255),
	price_unit FLOAT,
    unit_vol VARCHAR(255),
    price_ce TINYINT,
    PRIMARY KEY (product_id),
    FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id)
);
```
### IV. Wedding Vendors
### Inserting Values (Hair & Makeup Department)
```
#inserting for Vendor table until row 2539
/* INSERT QUERY NO: 1 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_01', 'hair and makeup', 'shineforth salon', 'oakland', 1, 50, 5104598967, '"""https://shineforth.com/"""'
);

/* INSERT QUERY NO: 2 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_02', 'hair and makeup', 'stylebee', 'san francisco', 0, 47, 4156940570, '"""https://www.stylebee.com/"""'
);

/* INSERT QUERY NO: 3 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_03', 'hair and makeup', 'bfab bridal wedding hairstylist makeup artist', 'santa clara', 0, 47, 6505173221, '"""https://www.bfabmobile.com/"""'
);

/* INSERT QUERY NO: 4 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_04', 'hair and makeup', 'beauty by rosheen', 'san jose', 0, 50, 4082502015, '"""http://www.rosheenahmadi.com/"""'
);

/* INSERT QUERY NO: 5 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_05', 'hair and makeup', 'maia beauty', 'hillsborough', 0, 50, 6508889676, '"""https://paulamaiabeauty.com/"""'
);

/* INSERT QUERY NO: 6 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_06', 'hair and makeup', 'phi phi yvonne', 'san jose', 0, 50, 4086097705, '"""https://www.phiphiyvonne.com/"""'
);

/* INSERT QUERY NO: 7 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_07', 'hair and makeup', 'polish and glo', 'berkeley', 1, 50, 4156042427, '"""https://polishandglo.com/"""'
);

/* INSERT QUERY NO: 8 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_08', 'hair and makeup', 'shannon le', 'san francisco', 0, 50, 4089311387, '"""https://www.shannonle.com/"""'
);

/* INSERT QUERY NO: 9 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_09', 'hair and makeup', 'beyond beauty inc.', 'san francisco', 0, 48, 5108286056, '"""https://www.beyondbeautyinc.com/"""'
);

/* INSERT QUERY NO: 10 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_10', 'hair and makeup', 'lindsay bauman', 'hayward', 0, 0, 6509182277, '"""https://www.lindsaybauman.com/"""'
);

/* INSERT QUERY NO: 11 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_11', 'hair and makeup', 'the pretty committee', 'oakland', 1, 49, 6162021273, '"""https://prettycommitteebeauty.com/"""'
);

/* INSERT QUERY NO: 12 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_12', 'hair and makeup', 'wow pretty makeup and hair agency', 'burlingame', 1, 48, 4158715512, '"""https://www.jiracouture.com/"""'
);

/* INSERT QUERY NO: 13 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_13', 'hair and makeup', 'tamramarie artistry', 'oakland', 0, 50, 5103333809, '"""https://www.tamramarie.com/"""'
);

/* INSERT QUERY NO: 14 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_14', 'hair and makeup', 'rachel makeup artist', 'sunnyvale', 0, 0, 4086747257, '"""https://www.rachelxmakeup.com/"""'
);

/* INSERT QUERY NO: 15 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_15', 'hair and makeup', 'linh artistry', 'san jose', 0, 50, 6505173221, '"""https://www.linhartistry.com/"""'
);

/* INSERT QUERY NO: 16 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_16', 'hair and makeup', 'primadonnamakeover hair  makeup  lipstick mix bar', 'san francisco', 0, 50, 4159875946, '"""https://www.primadonnamakeover.com/"""'
);

/* INSERT QUERY NO: 17 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_17', 'hair and makeup', 'moderne beauty', 'san jose', 0, 49, 4085492946, '"""https://www.instagram.com/modernebeauty/"""'
);

/* INSERT QUERY NO: 18 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_18', 'hair and makeup', 'last looks by tiffani', 'san leandro', 0, 50, 5103160859, '"""https://www.lastlooksbytiffani.com/"""'
);

/* INSERT QUERY NO: 19 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_19', 'hair and makeup', 'status salon', 'livermore', 0, 50, 9259808479, '"""https://statusalon.com/"""'
);

/* INSERT QUERY NO: 20 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_20', 'hair and makeup', 'beauty by pace', 'burlingame', 0, 50, 0, '"""https://www.beautybypace.com/"""'
);

/* INSERT QUERY NO: 21 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_21', 'hair and makeup', 'beauty by jasmine yin', 'san francisco', 0, 50, 0, '"""https://beautybyjasmineyin.com/"""'
);

/* INSERT QUERY NO: 22 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_22', 'hair and makeup', 'fox and belle salon', 'berkeley', 0, 50, 5107051582, '"""https://www.foxandbelle.com/"""'
);

/* INSERT QUERY NO: 23 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_23', 'hair and makeup', 'j larose beauty', 'san jose', 0, 50, 4159875946, '"""https://jlarosebeauty.com/"""'
);

/* INSERT QUERY NO: 24 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_24', 'hair and makeup', 'beauty by christine', 'santa clara', 1, 50, 4088745666, '"""https://www.teambridalbeauty.com/"""'
);

/* INSERT QUERY NO: 25 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_25', 'hair and makeup', 'misfit beauty', 'lafayette', 1, 0, 9165383179, '"""https://mbeautybyangelique.com/"""'
);

/* INSERT QUERY NO: 26 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_26', 'hair and makeup', 'stellakim artistry', 'san francisco', 1, 50, 4153501265, '"""https://www.stellakimartistry.com/"""'
);

/* INSERT QUERY NO: 27 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_27', 'hair and makeup', 'kim baker beauty', 'san jose', 1, 50, 4085026839, '"""https://www.kimbakerbeauty.com/"""'
);

/* INSERT QUERY NO: 28 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_28', 'hair and makeup', 'the hair studio', 'burlingame', 0, 50, 6508679190, '"""http://yourhairheaven.com/"""'
);

/* INSERT QUERY NO: 29 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_29', 'hair and makeup', 'la bae artistry', 'san carlos', 0, 50, 5108287146, '"""https://www.labaeartistry.com/"""'
);

/* INSERT QUERY NO: 30 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_30', 'hair and makeup', 'leece stylz by alicia lau', 'san francisco', 1, 50, 4155086030, '"""https://leecestylz.com/"""'
);

/* INSERT QUERY NO: 31 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_31', 'hair and makeup', 'nicoles vanity', 'san jose', 1, 50, 4086747257, '"""https://www.nicolesvanity.com/"""'
);

/* INSERT QUERY NO: 32 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_32', 'hair and makeup', 'sephora', 'san francisco', 0, 0, 4156440329, '"""https://www.sephora.com/happening/stores/san-francisco-centre"""'
);

/* INSERT QUERY NO: 33 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_33', 'hair and makeup', 'studio cathy hm beauty', 'san jose', 0, 50, 4088912976, '"""https://studiocathyhm.com/"""'
);

/* INSERT QUERY NO: 34 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_34', 'hair and makeup', 'sun beauty studio', 'fremont', 1, 50, 6508633296, '"""http://www.sunbeautystudio.com/"""'
);

/* INSERT QUERY NO: 35 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'hmu_35', 'hair and makeup', 'lash out loud beauty bar', 'los gatos', 0, 50, 4084492394, '"""https://lashoutloudbeautybar.com/"""'
);
```
### Inserting Values (Rental Department)
```
/* INSERT QUERY NO: 36 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_01', 'rental', '2 friends events', 'gilroy', 0, 50, 4088075457, '"""https:/www.2friendsevents.com"""'
);

/* INSERT QUERY NO: 37 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_02', 'rental', 'abbey party rents sf', 'daly city', 1, 47, 4157156900, '"""https://www.abbeyrentssf.com"""'
);

/* INSERT QUERY NO: 38 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_03', 'rental', 'all seasons event rental', 'redwood', 1, 45, 6503694419, '"""https://www.allseasonseventrentals.com"""'
);

/* INSERT QUERY NO: 39 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_04', 'rental', 'am party rentals', 'san leandro', 1, 48, 6503631050, '"""https://ampartyrentals.com"""'
);

/* INSERT QUERY NO: 40 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_05', 'rental', 'bb event productions', 'hayward', 1, 0, 5105665677, '"""https://www.bbeventproductions.com"""'
);

/* INSERT QUERY NO: 41 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_06', 'rental', 'bee amazing events', 'pittsburg', 0, 50, 9253317330, '"""https://biancajavierevents.com"""'
);

/* INSERT QUERY NO: 42 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_07', 'rental', 'bright events rentals', 'brisbane', 1, 46, 4155700470, '"""https://www.bright.com"""'
);

/* INSERT QUERY NO: 43 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_08', 'rental', 'celebrations of marin', 'san alselmo', 0, 50, 4152617346, '"""https://www.celebrationsofmarin.com"""'
);

/* INSERT QUERY NO: 44 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_09', 'rental', 'chairs for affairs', 'concord', 1, 49, 9253701300, '"""https://www.chairsforaffairs.com"""'
);

/* INSERT QUERY NO: 45 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_10', 'rental', 'fine linen creations', 'san jose', 1, 50, 4082169512, '"""https://www.finelinencreation.com"""'
);

/* INSERT QUERY NO: 46 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_11', 'rental', 'g n event rental inc', 'san jose', 1, 50, 4088594080, '"""https://gneventrental.com"""'
);

/* INSERT QUERY NO: 47 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_12', 'rental', 'good events', 'hayward', 1, 50, 5102447921, '"""https://goodevents.com"""'
);

/* INSERT QUERY NO: 48 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_13', 'rental', 'hensley event resources', 'brisbane', 0, 41, 6506927007, '"""https://www.hensleyeventresources.com"""'
);

/* INSERT QUERY NO: 49 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_14', 'rental', 'janettes events', 'san francisco', 1, 50, 4157946704, '"""https://www.eventsbythebay.com"""'
);

/* INSERT QUERY NO: 50 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_15', 'rental', 'lux event rentals and design', 'antioch', 1, 50, 9253656055, '"""https://luxeventrentalsanddesign.com"""'
);

/* INSERT QUERY NO: 51 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_16', 'rental', 'piedmont party rentals', 'oakland', 1, 48, 5106323400, '"""https://www.piedmontparty.com"""'
);

/* INSERT QUERY NO: 52 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_17', 'rental', 'pleasanton rentals inc', 'livermore', 1, 45, 9254680624, '"""https://www.pleasantonrentals.com"""'
);

/* INSERT QUERY NO: 53 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_18', 'rental', 'posh balloon studio ', 'palo alto', 1, 50, 8889287674, '"""https://www.poshballoonstudio.com"""'
);

/* INSERT QUERY NO: 54 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_19', 'rental', 'rsvp decor', 'san joaquin valley', 1, 50, 5105023900, '"""https://www.rsvpdecor.com"""'
);

/* INSERT QUERY NO: 55 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_20', 'rental', 'special events rental', 'livermore', 0, 35, 8005400657, '"""https://specialeventsrental.com"""'
);

/* INSERT QUERY NO: 56 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_21', 'rental', 'stuart rental', 'san jose', 1, 38, 8004000543, '"""https://www.stuartrental.com"""'
);

/* INSERT QUERY NO: 57 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_22', 'rental', 'sweet little details', 'oakley', 0, 50, 9253917954, '"""https://sweetlittledetails.com"""'
);

/* INSERT QUERY NO: 58 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_23', 'rental', 'sweet wonders candy buffet and event design', 'san francisco', 0, 0, 5103429418, '"""https://www.sweetwondersdesign.com"""'
);

/* INSERT QUERY NO: 59 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_24', 'rental', 'terra amico', 'saint martin', 0, 50, 4089871546, '"""https://www.terraamico.com"""'
);

/* INSERT QUERY NO: 60 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ren_25', 'rental', 'zephyr tents', 'berkeley', 1, 50, 5105598694, '"""https://zephyrtents.com"""'
);
```
### Inserting Values (Catering Department)
```
/* INSERT QUERY NO: 61 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_01', 'catering', 'hola paella', 'cupertino', 0, 50, 4082030205, ''
);

/* INSERT QUERY NO: 62 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_02', 'catering', 'customized catering', 'san jose', 0, 50, 5104270423, ''
);

/* INSERT QUERY NO: 63 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_03', 'catering', 'steel smokin bbq and catering', 'pleasant hill', 0, 50, 9253233040, ''
);

/* INSERT QUERY NO: 64 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_04', 'catering', 'go streatery', 'bay area', 0, 50, 5107779777, ''
);

/* INSERT QUERY NO: 65 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_05', 'catering', 'slider shack', 'san francisco', 0, 50, 4156722902, ''
);

/* INSERT QUERY NO: 66 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_06', 'catering', 'bi-rite market', 'san francisco', 0, 50, 4152419760, ''
);

/* INSERT QUERY NO: 67 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_07', 'catering', 'caterman catering', 'san jose', 1, 48, 4084418719, ''
);

/* INSERT QUERY NO: 68 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_08', 'catering', 'la mediterranee catering', 'san francisco', 1, 50, 4159212956, ''
);

/* INSERT QUERY NO: 69 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_09', 'catering', 'miss neetas ice cream parlor', 'hayward', 0, 50, 5103047716, ''
);

/* INSERT QUERY NO: 70 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_10', 'catering', 'dd catering inc', 'sunnyvale', 1, 50, 4087171131, ''
);

/* INSERT QUERY NO: 71 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_11', 'catering', 'basilico italian catering', 'concord', 0, 50, 9258187374, ''
);

/* INSERT QUERY NO: 72 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_12', 'catering', 'total wine and more', 'san francisco', 1, 47, 0, ''
);

/* INSERT QUERY NO: 73 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_13', 'catering', 'elmas catering and events', 'san jose', 0, 0, 4083750538, ''
);

/* INSERT QUERY NO: 74 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_14', 'catering', 'nob hill catering inc', 'san carlos', 0, 0, 6505968008, ''
);

/* INSERT QUERY NO: 75 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_15', 'catering', 'sample this fine catering', 'half moon bay', 0, 0, 6507262044, ''
);

/* INSERT QUERY NO: 76 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_16', 'catering', 'fire 4 hire catering', 'gilroy', 1, 49, 4087103716, ''
);

/* INSERT QUERY NO: 77 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_17', 'catering', 'events etc', 'south san francisco', 1, 0, 6509521110, ''
);

/* INSERT QUERY NO: 78 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_18', 'catering', 'just relish', 'berkeley', 1, 0, 5109101174, ''
);

/* INSERT QUERY NO: 79 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_19', 'catering', 'mccalls catering', 'san francisco', 1, 50, 4155528550, '"""https://mccallssf.com"""'
);


/* INSERT QUERY NO: 80 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_20', 'catering', 'quake catering', 'san jose', 0, 0, 4087689220, '"""https://www.quakecatering.com"""'
);

/* INSERT QUERY NO: 81 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_21', 'catering', 'blue heron catering inc', 'castro valley', 1, 48, 5105330781, ''
);

/* INSERT QUERY NO: 82 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_22', 'catering', 'a successful event catering', 'san jose', 1, 0, 4082120322, ''
);

/* INSERT QUERY NO: 83 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_23', 'catering', 'fogcutter', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 84 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_24', 'catering', 'wylder space', 'scotts valley', 1, 50, 8317047494, ''
);

/* INSERT QUERY NO: 85 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_25', 'catering', 'melons catering and events', 'south san francisco', 1, 43, 6505831756, ''
);

/* INSERT QUERY NO: 86 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_26', 'catering', 'global gourmet catering', 'san francisco', 1, 50, 4157010001, ''
);

/* INSERT QUERY NO: 87 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_27', 'catering', 'fraiche catering', 'san francisco', 1, 50, 9256832800, ''
);

/* INSERT QUERY NO: 88 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_28', 'catering', 'flour and branch', 'san francisco', 1, 50, 4152338775, ''
);

/* INSERT QUERY NO: 89 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_29', 'catering', 'handheld catering', 'santa clara', 1, 39, 4086924782, ''
);

/* INSERT QUERY NO: 90 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_30', 'catering', 'paula leduc fine catering and events', 'emeryville', 1, 34, 5105477825, ''
);

/* INSERT QUERY NO: 91 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_31', 'catering', 'san franciscos hometown creamery', 'san francisco', 0, 50, 4156824977, ''
);

/* INSERT QUERY NO: 92 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_32', 'catering', 'nutrigastro', 'san francisco', 1, 45, 8554125099, ''
);

/* INSERT QUERY NO: 93 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_33', 'catering', 'the homestead fine catering', 'santa clara', 0, 50, 4086385300, ''
);

/* INSERT QUERY NO: 94 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_34', 'catering', 'midsummer kitchen', 'concord', 1, 50, 9258581049, ''
);

/* INSERT QUERY NO: 95 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_35', 'catering', 'bsc catering', 'walnut creek', 1, 45, 9254481661, ''
);

/* INSERT QUERY NO: 96 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_36', 'catering', 'trumpetvine catering and events', 'paso robles', 1, 49, 5108487268, ''
);

/* INSERT QUERY NO: 97 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_37', 'catering', 'above and beyond catering', 'san francisco', 0, 25, 4153084825, ''
);

/* INSERT QUERY NO: 98 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_38', 'catering', 'eat play events and catering', 'south bay', 1, 50, 5102748335, ''
);

/* INSERT QUERY NO: 99 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_39', 'catering', 'lre catering', 'san francisco', 1, 35, 4155221417, ''
);

/* INSERT QUERY NO: 100 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_40', 'catering', 'tortellino bologna', 'san francisco', 1, 0, 6282357927, ''
);

/* INSERT QUERY NO: 101 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_41', 'catering', 'crystal springs catering', 'redwood city', 1, 48, 6503655070, ''
);

/* INSERT QUERY NO: 102 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_42', 'catering', 'svetlana catering and events', 'emeryville', 1, 49, 4153140782, ''
);

/* INSERT QUERY NO: 103 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_43', 'catering', 'checkers catering and special events', 'livermore', 1, 49, 9254575947, ''
);

/* INSERT QUERY NO: 104 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_44', 'catering', 'amber india restaurants', 'mountain view', 0, 40, 6509687511, ''
);

/* INSERT QUERY NO: 105 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_45', 'catering', 'culinary eye catering and events', 'san francisco', 1, 49, 4158241225, ''
);

/* INSERT QUERY NO: 106 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_46', 'catering', 'carrie dove catering and events', 'oakland', 1, 50, 4154609995, ''
);

/* INSERT QUERY NO: 107 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_47', 'catering', 'cater2usf', 'san francisco', 1, 50, 4159485203, ''
);

/* INSERT QUERY NO: 108 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_48', 'catering', 'miller and lux', 'san francisco', 1, 0, 4158726694, '"""https://millerandluxrestaurant.com"""'
);

/* INSERT QUERY NO: 109 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_49', 'catering', 'taste catering and event planning', 'napa', 0, 44, 4155506464, ''
);

/* INSERT QUERY NO: 110 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'cat_50', 'catering', 'stock and bones', 'san francisco', 1, 0, 4154055316, ''
);
```
### Inserting Values (Venues Department)
```
/* INSERT QUERY NO: 111 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_01', 'venues', 'the bridges golf club', 'san ramon ', 1, 48, 9257354253, '"""https://www.thebridgesgolf.com/-wedding-overview"""'
);

/* INSERT QUERY NO: 112 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_02', 'venues', 'viaggio estate and winery', 'acampo ', 1, 41, 2093330484, '"""https://www.viaggiowinery.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 113 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_03', 'venues', 'leal vineyards', 'hollister ', 1, 48, 8316361023, '"""https://www.liveloveleal.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 114 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_04', 'venues', 'oceano hotel and spa', 'half moon bay ', 1, 49, 6507265400, '"""https://oceanohalfmoonbay.com/weddings/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 115 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_05', 'venues', 'log cabin at the presidio', 'san francisco ', 1, 48, 8669663009, '"""https://www.wedgewoodweddings.com/logcabin?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 116 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_06', 'venues', 'callippe preserve', 'pleasanton ', 1, 49, 9254266666, '"""https://www.playcallippe.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 117 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_07', 'venues', 'the garden walnut creek', 'walnut creek ', 1, 0, 9259485599, '"""https://www.thegardenwc.com/"""'
);

/* INSERT QUERY NO: 118 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_08', 'venues', 'the terrace room at lake merritt', 'oakland ', 0, 49, 5109033580, '"""https://www.theterraceroomevents.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 119 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_09', 'venues', 'berkeley country club', 'el cerrito ', 1, 50, 5108061037, '"""https://www.berkeleycountryclub.com/weddings-events?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 120 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_10', 'venues', 'pinstripes san mateo', 'san mateo ', 1, 50, 6507814721, '"""https://pinstripes.com/san-mateo/private-events/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 121 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_11', 'venues', 'the club at castlewood', 'pleasanton ', 1, 47, 9254852237, '"""https://www.castlewoodcc.org/"""'
);

/* INSERT QUERY NO: 122 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_12', 'venues', 'berkeley city club', 'berkeley ', 0, 44, 5108091953, '"""https://www.berkeleycityclub.com/weddings.htm"""'
);

/* INSERT QUERY NO: 123 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_13', 'venues', 'marines memorial club and hotel union square', 'san francisco ', 0, 48, 4158309133, '"""https://marinesmemorial.org/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 124 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_14', 'venues', 'rancho nicasio', 'nicasio ', 1, 50, 4152646859, '"""https://www.ranchonicasio.com/weddings/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 125 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_15', 'venues', 'fairview metropolitan', 'oakland ', 1, 46, 5105695555, '"""https://www.fairviewevents.com/metropolitan/info-metro"""'
);

/* INSERT QUERY NO: 126 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_16', 'venues', 'corinthian yacht club', 'tiburon ', 1, 50, 4154354771, '"""https://www.cyc.org/catering?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 127 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_17', 'venues', 'coyote creek golf club', 'san jose ', 1, 50, 4084631800, '"""https://coyotecreekgolf.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 128 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_18', 'venues', 'the purple orchid wine country resort and spa', 'livermore ', 1, 46, 9256068855, '"""https://www.purpleorchid.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 129 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_19', 'venues', 'yin ranch', 'vacaville ', 1, 20, 7077708695, '"""https://yinranch.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 130 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_20', 'venues', 'the university club of san francisco', 'san francisco ', 0, 50, 4158007517, '"""https://www.uclubsf.org/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 131 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_21', 'venues', 'san jose marriott', 'san jose ', 1, 50, 4082784428, '"""https://www.marriott.com/en-us/hotels/sjcsj-san-jose-marriott/overview/"""'
);

/* INSERT QUERY NO: 132 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_22', 'venues', 'beacon grand', 'san francisco ', 0, 49, 4153927755, '"""https://www.beacongrand.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 133 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_23', 'venues', 'oakhurst county club', 'clayton ', 1, 48, 9256729737, '"""https://www.oakhurstcc.com/weddings-events/pricing?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 134 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_24', 'venues', 'hotel nia autograph collection', 'menlo park ', 1, 50, 6503043747, '"""https://www.hotelnia.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 135 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_25', 'venues', 'mammoth mountain ski area', 'mammoth lakes ', 1, 48, 7609142561, '"""https://www.mammothmountain.com/discover-mammoth/groups-and-weddings/weddings?utm_medium=referral&utm_source=theknot.com&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 136 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_26', 'venues', 'mountain house estate', 'cloverdale ', 1, 50, 7703851410, '"""https://mountainhouseestate.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 137 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_27', 'venues', 'roaring camp railroads', 'felton ', 1, 46, 8313354484, '"""https://roaringcamp.com/weddings"""'
);

/* INSERT QUERY NO: 138 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_28', 'venues', 'curiodyssey', 'san mateo ', 1, 49, 6503407565, '"""https://curiodyssey.org/book-an-event/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 139 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_29', 'venues', 'babette', 'berkeley ', 1, 50, 4157133591, '"""https://www.babettecafe.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 140 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_30', 'venues', 'stanford faculty club', 'stanford ', 1, 38, 6507254467, '"""https://facultyclub.stanford.edu/weddings?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 141 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_31', 'venues', 'the monk ranch', 'dixon ', 0, 50, 5302192324, ''
);

/* INSERT QUERY NO: 142 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_32', 'venues', 'green oaks creek farm', 'pescadero ', 1, 50, 6508791009, '"""https://www.greenoakswedding.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 143 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_33', 'venues', 'hilton san jose', 'san jose ', 1, 40, 4082872100, '"""https://www.hilton.com/en/hotels/sjcshhf-hilton-san-jose/"""'
);

/* INSERT QUERY NO: 144 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_34', 'venues', 'university of redlands marin campus', 'san anselmo ', 1, 47, 4154512843, '"""https://www.redlands.edu/marin-events/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 145 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_35', 'venues', 'grand hyatt at sfo', 'san francisco ', 0, 0, 6504521234, '"""https://www.hyatt.com/en-us/hotel/california/grand-hyatt-at-sfo/sfogh?src=corp_lclb_gmb_seo_nam_sfogh"""'
);

/* INSERT QUERY NO: 146 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_36', 'venues', 'unity in marin weddings', 'novato ', 0, 30, 4154755000, '"""https://unityinmarin.org/rentals-weddings/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 147 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_37', 'venues', 'regale winery and vineyards', 'los gatos ', 1, 50, 4083532500, '"""https://www.regalewine.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 148 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_38', 'venues', 'swedish american hall', 'san francisco ', 0, 0, 4158267005, '"""https://swedishamericanhall.com/"""'
);

/* INSERT QUERY NO: 149 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_39', 'venues', 'elliston vineyards', 'sunol ', 1, 47, 9258622377, '"""https://elliston.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 150 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_40', 'venues', 'solage calistoga', 'calistoga ', 1, 40, 7072260881, '"""https://aubergeresorts.com/solage/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 151 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_41', 'venues', 'riggers loft wine company', 'richmond ', 1, 50, 5086928414, '"""https://www.riggersloftwine.com/"""'
);

/* INSERT QUERY NO: 152 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_42', 'venues', 'blanco urban venue', 'san jose ', 1, 50, 4084772424, '"""https://blancourbanvenue.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 153 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_43', 'venues', 'brownstone gardens', 'oakley ', 1, 48, 9254184532, '"""https://www.brownstonegardens.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 154 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_44', 'venues', 'villa ragusa', 'campbell ', 0, 46, 4083641900, '"""https://villaragusa.com/"""'
);

/* INSERT QUERY NO: 155 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_45', 'venues', 'sf museum of modern art', 'san francisco ', 1, 0, 4159151777, '"""https://www.sfmoma.org/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 156 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_46', 'venues', 'cavallo point the lodge at the golden gate', 'sausalito ', 1, 38, 4153394731, '"""https://www.cavallopoint.com/gatherings/weddings/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 157 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_47', 'venues', 'webster hall sf', 'san francisco ', 0, 0, 4157156900, '"""https://www.websterhallsf.com/?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 158 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_48', 'venues', 'the pearl', 'san francisco ', 1, 46, 7069848004, '"""https://www.theknot.com/marketplace/the-pearl-columbus-ga-2078016"""'
);

/* INSERT QUERY NO: 159 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_49', 'venues', 'half moon bay golf links', 'half moon bay ', 1, 45, 6507122250, '"""https://www.halfmoonbaygolf.com/weddings?utm_source=theknot.com&utm_medium=referral&utm_campaign=theknot"""'
);

/* INSERT QUERY NO: 160 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'ven_50', 'venues', 'omni san francisco hotel', 'san francisco ', 0, 45, 4152733014, '"""https://www.omnihotels.com/hotels/san-francisco"""'
);
```
### Inserting Values (Jewelry Department)
```
/* INSERT QUERY NO: 161 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'jwl_01', 'jewelry', 'brilliant earth', 'san francisco', 1, 41, 8006910952, '"""https://www.brilliantearth.com/"""'
);

/* INSERT QUERY NO: 162 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'jwl_02', 'jewelry', 'altana marie', 'oakland', 0, 50, 5104889960, '"""https://www.altanamarie.com/"" "'
);

/* INSERT QUERY NO: 163 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'jwl_03', 'jewelry', 'sausolito jewelers', 'sausolito', 1, 50, 4153312100, '"""https://www.sausolitojewelers.com/"""'
);

/* INSERT QUERY NO: 164 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'jwl_04', 'jewelry', 'diamondere', 'palo alto', 0, 49, 8444000065, '"""https://www.diamondere.com/"""'
);
```
### Inserting Values (Flowers Department)
```
/* INSERT QUERY NO: 165 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_01', 'flowers', 'flowers of the valley', 'san francisco', 1, 0, 4159700579, '"""https://flowersofthevalleysf.com/"""'
);

/* INSERT QUERY NO: 166 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_02', 'flowers', 'eventscapes inc', 'watsonville', 0, 0, 8317282980, '"""https://eventscapesinc.com/"""'
);

/* INSERT QUERY NO: 167 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_03', 'flowers', 'wild iris floral and botanical', 'felton', 1, 0, 8312780151, '"""https://www.wildirisfloral.com/"""'
);

/* INSERT QUERY NO: 168 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_04', 'flowers', 'hills flowers and events', 'san jose', 1, 0, 4082956735, '"""https://hillsflowers.net/"""'
);

/* INSERT QUERY NO: 169 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_05', 'flowers', 'florally fleurish', 'alameda', 1, 0, 5107546087, '"""https://www.florallyfleurish.com/"""'
);

/* INSERT QUERY NO: 170 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_06', 'flowers', 'michaelas flower shop', 'palo alto', 1, 0, 6503215390, '"""https://www.michaelasflowershop.com/"""'
);

/* INSERT QUERY NO: 171 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_07', 'flowers', 'expressions floral', 'gilroy', 0, 0, 4088428400, '"""https://www.expressionsfloralgilroy.com/"""'
);

/* INSERT QUERY NO: 172 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_08', 'flowers', 'church street flowers', 'san francisco', 1, 0, 4154343427, '"""https://churchstreetflowers.com/flower-arrangements/"""'
);

/* INSERT QUERY NO: 173 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_09', 'flowers', 'floret design', 'san jose', 1, 0, 4086557399, '"""https://floretdesignstudio.com/"""'
);

/* INSERT QUERY NO: 174 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_10', 'flowers', 'fleurish with flowers', 'san francisco', 0, 0, 6507593527, '"""https://www.fleurishwithflowers.com/"""'
);

/* INSERT QUERY NO: 175 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_11', 'flowers', 'sweetheart florist and trading llc', 'san francisco', 1, 0, 4153923098, '"""https://sweetheartfloristsf.com/"""'
);

/* INSERT QUERY NO: 176 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_12', 'flowers', 'joellen pope weddings', 'napa', 1, 0, 7073378349, '"""https://www.joellenpopeweddings.com/"""'
);

/* INSERT QUERY NO: 177 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_13', 'flowers', 'niche and nook flowers', 'san francisco', 1, 0, 4157963281, '"""https://www.nicheandnookflowers.com/"""'
);

/* INSERT QUERY NO: 178 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_14', 'flowers', 'burns florist', 'san rafael', 0, 0, 4154543706, '"""https://www.burnsflowershop.com/"""'
);

/* INSERT QUERY NO: 179 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_15', 'flowers', 'thistledown designs', 'pleasant hill', 1, 0, 9253305415, '"""https://thistledowndesigns.wordpress.com/"""'
);

/* INSERT QUERY NO: 180 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_16', 'flowers', 'hana bloom floral design', 'san jose', 1, 0, 4086415138, '"""https://www.hanabloomfloral.com/"""'
);

/* INSERT QUERY NO: 181 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_17', 'flowers', 'not just flowers', 'san francisco', 1, 0, 4158643134, '"""https://notjustflowerscastro.com/san-francisco-florist-flower-delivery/"""'
);

/* INSERT QUERY NO: 182 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_18', 'flowers', 'floral design studio', 'fremont', 1, 0, 5106563010, '"""https://www.floraldesignstudiofremont.com/"""'
);

/* INSERT QUERY NO: 183 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_19', 'flowers', 'dannas flowers', 'san jose', 0, 0, 4087237111, '"""https://www.dannasflowers.com/"""'
);

/* INSERT QUERY NO: 184 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_20', 'flowers', 'mandy scott events', 'san francisco', 1, 0, 4154415973, '"""https://www.mandyscottevents.com/index.html"""'
);

/* INSERT QUERY NO: 185 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_21', 'flowers', 'franks garden florist', 'gilroy', 1, 0, 4088475888, '"""https://www.weddingwire.com/biz/franks-garden-florist-gilroy/8e34e909069623a6.html"""'
);

/* INSERT QUERY NO: 186 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_22', 'flowers', 'absolute elegance floral', 'daly city', 0, 0, 4155770044, '"""https://www.aefloralsf.com/"""'
);

/* INSERT QUERY NO: 187 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_23', 'flowers', 'lulus house of flowers', 'san jose', 1, 0, 4084188604, '"""https://www.houseoflulu.org/"""'
);

/* INSERT QUERY NO: 188 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_24', 'flowers', 'c and m fleuri', 'san jose', 0, 0, 4089100254, '"""https://cmfleuri.com/"""'
);

/* INSERT QUERY NO: 189 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_25', 'flowers', 'flower divas  inc', 'santa clara', 1, 0, 4088912488, '"""https://flowerdivas.com/"""'
);

/* INSERT QUERY NO: 190 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_26', 'flowers', 'silks are forever', 'san francisco', 1, 0, 8004511598, '"""https://www.silksareforever.com/?gad_source=1&gclid=CjwKCAiA1MCrBhAoEiwAC2d64bzWopCGjm_qm9bi5JklFyyRKgEvn8rp9aeog2NoNMADM-rguIUvBhoC4MgQAvD_BwE"""'
);

/* INSERT QUERY NO: 191 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_27', 'flowers', 'rafael florist', 'san rafael', 1, 0, 4154574712, '"""https://rafaelflorist.com/"""'
);

/* INSERT QUERY NO: 192 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_28', 'flowers', 'just a flower guy', 'san francisco', 0, 0, 5104954262, '"""https://www.weddingwire.com/biz/just-a-flower-guy-orinda/cd6699b4d2de462b.html"""'
);

/* INSERT QUERY NO: 193 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_29', 'flowers', 'le fleur d', 'hercules', 1, 0, 5109644622, '"""https://www.lefleurd.com/"""'
);

/* INSERT QUERY NO: 194 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_30', 'flowers', 'cypress flower design', 'santa clara', 0, 0, 4086936413, '"""https://cypressflowersdesign.com/"""'
);

/* INSERT QUERY NO: 195 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_31', 'flowers', 'the prickly petal flower co', 'san jose', 1, 0, 4083208241, '"""https://www.weddingwire.com/biz/the-prickly-petal-flower-co-san-jose/9629875c9faf0636.html"""'
);

/* INSERT QUERY NO: 196 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_32', 'flowers', 'dragonfly floristic', 'cupertino', 0, 0, 6692929757, '"""https://www.dragonflyfloristic.com/"""'
);

/* INSERT QUERY NO: 197 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_33', 'flowers', 'diyari wedding', 'livermore', 1, 0, 9252907116, '"""https://www.weddingwire.com/biz/diyari-wedding-livermore/360a750cc3bdca0f.html"""'
);

/* INSERT QUERY NO: 198 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_34', 'flowers', 'bloomsters', 'san jose', 0, 0, 4082685518, '"""https://www.bloomsters.com/"""'
);

/* INSERT QUERY NO: 199 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_35', 'flowers', 'mindy rosenberg design', 'burlingame', 0, 0, 8478262184, '"""https://www.mindyrosenberg.com/"""'
);

/* INSERT QUERY NO: 200 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_36', 'flowers', 'fleurish ca', 'oakland', 1, 0, 5109172006, '"""https://www.fleurishca.com/"""'
);

/* INSERT QUERY NO: 201 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_37', 'flowers', 'rosies and posies florist', 'campbell', 0, 0, 4083784046, '"""https://www.rosiesandposiesca.com/"""'
);

/* INSERT QUERY NO: 202 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_38', 'flowers', 'flowers valley', 'santa clara', 0, 0, 7078936686, '"""https://www.theflowersvalley.com/"""'
);

/* INSERT QUERY NO: 203 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_39', 'flowers', 'flowers of joy', 'concord', 1, 0, 9258521740, '"""https://flowersofjoy.biz/"""'
);

/* INSERT QUERY NO: 204 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_40', 'flowers', 'the plan it duo', 'corte madera', 1, 0, 4156061658, '"""https://www.weddingwire.com/biz/the-plan-it-duo-event-planning-corte-madera/5f4250b787cb42c1.html"""'
);

/* INSERT QUERY NO: 205 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_41', 'flowers', 'scarborough affairs', 'san luis obispo', 1, 0, 9728009717, '"""https://www.weddingwire.com/biz/scarborough-affair/ea478e67e1336712.html"""'
);

/* INSERT QUERY NO: 206 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'flo_42', 'flowers', 'a loves in bloom', 'martinez', 1, 0, 9254086702, '"""http://alovesinbloom.com/"""'
);
```
### Inserting Values (Dress & Attire Department)
```
/* INSERT QUERY NO: 207 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_01', 'dress and attire', 'casablanca bridal', 'san francisco', 0, 0, 0, '"""https://www.casablancabridal.com/"""'
);

/* INSERT QUERY NO: 208 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_02', 'dress and attire', 'allure bridal', 'online', 0, 0, 0, '"""https://allurebridals.com/"""'
);

/* INSERT QUERY NO: 209 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_03', 'dress and attire', 'stacees', 'online', 0, 0, 0, '"""https://www.stacees.com/"""'
);

/* INSERT QUERY NO: 210 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_04', 'dress and attire', 'misaac', 'online', 0, 0, 0, '"""https://www.missacc.com"""'
);

/* INSERT QUERY NO: 211 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_05', 'dress and attire', 'jj house', 'online', 0, 0, 0, '"""https://www.jjshouse.com/"""'
);

/* INSERT QUERY NO: 212 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_06', 'dress and attire', 'birdy grey', 'online', 1, 0, 0, '"""https://www.birdygrey.com/"""'
);

/* INSERT QUERY NO: 213 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_07', 'dress and attire', 'aw bridal', 'online', 0, 0, 0, '"""https://www.awbridal.com/"""'
);

/* INSERT QUERY NO: 214 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_08', 'dress and attire', 'madeline gardner', 'online', 0, 0, 0, '"""https://www.madelinegardnernewyork.com/"""'
);

/* INSERT QUERY NO: 215 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_09', 'dress and attire', 'dessy group', 'san francisco', 1, 0, 0, '"""https://dessy.com/"""'
);

/* INSERT QUERY NO: 216 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_10', 'dress and attire', 'christina wu celebration', 'online', 0, 0, 0, '"""https://www.christinawucollection.com/"""'
);

/* INSERT QUERY NO: 217 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_11', 'dress and attire', 'kennedy blue', 'online', 1, 0, 0, '"""https://www.kennedyblue.com/"""'
);

/* INSERT QUERY NO: 218 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_12', 'dress and attire', 'jasmine bridal', 'online', 0, 0, 0, '"""https://www.jasminebridal.com/"""'
);

/* INSERT QUERY NO: 219 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_13', 'dress and attire', 'mon cheri bridals', 'online', 0, 0, 0, '"""https://www.moncheribridals.com/"""'
);

/* INSERT QUERY NO: 220 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_14', 'dress and attire', 'men\'s warehouse', 'san francisco', 1, 0, 0, '"""https://www.menswearhouse.com/"""'
);

/* INSERT QUERY NO: 221 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_15', 'dress and attire', 'atelier munro', 'san francisco', 1, 0, 0, '"""https://ateliermunro.com/"""'
);

/* INSERT QUERY NO: 222 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_16', 'dress and attire', 'brunello cucinelli', 'san francisco', 1, 0, 0, '"""https://shop.brunellocucinelli.com/"""'
);

/* INSERT QUERY NO: 223 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'att_17', 'dress and attire', 'blacktux', 'online', 1, 0, 0, '"""https://theblacktux.com/"""'
);
```
### Inserting Values (Photo & Video Department)
```
/* INSERT QUERY NO: 224 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_01', 'photo and video', 'love genre films', 'sacramento', 1, 50, 0, ''
);

/* INSERT QUERY NO: 225 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_02', 'photo and video', 'apollo fotografie', 'san francisco', 1, 49, 0, ''
);

/* INSERT QUERY NO: 226 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_03', 'photo and video', '1pshot', 'belmont', 1, 50, 0, ''
);

/* INSERT QUERY NO: 227 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_04', 'photo and video', 'silver seas', 'sausalito', 1, 49, 0, ''
);

/* INSERT QUERY NO: 228 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_05', 'photo and video', 'amari productions', 'san diego', 1, 50, 0, ''
);

/* INSERT QUERY NO: 229 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_06', 'photo and video', 'skippy tv weddings', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 230 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_07', 'photo and video', 'avalanche film', 'los angeles', 1, 50, 0, ''
);

/* INSERT QUERY NO: 231 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_08', 'photo and video', 'danny rey films', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 232 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_09', 'photo and video', 'boundary visual media', 'san rafael', 1, 0, 0, ''
);

/* INSERT QUERY NO: 233 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_10', 'photo and video', 'inventive films', 'napa', 1, 50, 0, ''
);

/* INSERT QUERY NO: 234 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_11', 'photo and video', 'amora cinema', 'san luis obispo', 1, 50, 0, ''
);

/* INSERT QUERY NO: 235 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_12', 'photo and video', 'driftr films', 'pleasanton', 1, 50, 0, ''
);

/* INSERT QUERY NO: 236 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_13', 'photo and video', 'final frame studios', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 237 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_14', 'photo and video', 'brighter lights', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 238 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_15', 'photo and video', 'timeless tree weddings', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 239 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_16', 'photo and video', 'sasha photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 240 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_17', 'photo and video', 'peek media', 'berkeley', 1, 50, 0, ''
);

/* INSERT QUERY NO: 241 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_18', 'photo and video', 'lumitone photography and cinematography', 'san francisco', 1, 48, 0, ''
);

/* INSERT QUERY NO: 242 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_19', 'photo and video', 'hand in hand production', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 243 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_20', 'photo and video', 'matthew james ross photo and video', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 244 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_21', 'photo and video', 'maineline studios', 'san francisco', 1, 40, 0, ''
);

/* INSERT QUERY NO: 245 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_22', 'photo and video', 'privilege films', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 246 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_23', 'photo and video', 'modest reaction films', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 247 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_24', 'photo and video', 'hugo film co', 'carmel', 1, 50, 0, ''
);

/* INSERT QUERY NO: 248 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_25', 'photo and video', 'george street photo and video', 'san francisco', 1, 43, 0, ''
);

/* INSERT QUERY NO: 249 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_26', 'photo and video', 'christophe genty photography', 'napa', 1, 50, 0, ''
);

/* INSERT QUERY NO: 250 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_27', 'photo and video', 'trung hoang photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 251 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_28', 'photo and video', 'michael dadula photography', 'monterey', 1, 50, 0, ''
);

/* INSERT QUERY NO: 252 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_29', 'photo and video', 'honeystills photography', 'oakland', 1, 48, 0, ''
);

/* INSERT QUERY NO: 253 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_30', 'photo and video', 'annamae photo', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 254 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_31', 'photo and video', 'jenn justice photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 255 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_32', 'photo and video', 'the indi collective', 'morro bay', 1, 50, 0, ''
);

/* INSERT QUERY NO: 256 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_33', 'photo and video', 'kelli santos photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 257 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_34', 'photo and video', 'all on deck photos', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 258 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_35', 'photo and video', 'lucky devils creative', 'los angeles', 1, 40, 0, ''
);

/* INSERT QUERY NO: 259 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_36', 'photo and video', 'bydesign photo films', 'studio', 1, 48, 0, ''
);

/* INSERT QUERY NO: 260 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_37', 'photo and video', 'splashes of time photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 261 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_38', 'photo and video', 'jennifer mihalyi photography', 'sacramento', 1, 50, 0, ''
);

/* INSERT QUERY NO: 262 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_39', 'photo and video', 'hannah leigh llc', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 263 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_40', 'photo and video', 'alex pimentel photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 264 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_41', 'photo and video', 'alina roz photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 265 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_42', 'photo and video', 'fotogems', 'los gatos', 1, 50, 0, ''
);

/* INSERT QUERY NO: 266 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_43', 'photo and video', 'just in pix', 'san jose', 1, 0, 0, ''
);

/* INSERT QUERY NO: 267 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_44', 'photo and video', 'daelin waschke photography', 'san francisco', 1, 0, 0, ''
);

/* INSERT QUERY NO: 268 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_45', 'photo and video', 'forefront photography', 'san francisco', 1, 0, 0, ''
);

/* INSERT QUERY NO: 269 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_46', 'photo and video', 'julia goldberg photography', 'san francisco', 0, 50, 0, ''
);

/* INSERT QUERY NO: 270 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_47', 'photo and video', 'bailey w photography', 'san francisco', 1, 50, 0, ''
);

/* INSERT QUERY NO: 271 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_48', 'photo and video', 'romantic photographer', 'san rafael', 1, 0, 0, ''
);

/* INSERT QUERY NO: 272 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_49', 'photo and video', 'weddings by samuel', 'greenbrae', 1, 0, 0, ''
);

/* INSERT QUERY NO: 273 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'vid_50', 'photo and video', 'john leestma photography', 'petaluma', 1, 50, 0, ''
);
```
### Inserting Values (Invitations Department)
```
/* INSERT QUERY NO: 274 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_01', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/always-forever-wedding-invitations?v=9d977cc4-17aa-464d-a854-16bfef5eb335&q=100"""'
);

/* INSERT QUERY NO: 275 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_02', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/always-forever-wedding-invitations?v=9d977cc4-17aa-464d-a854-16bfef5eb335&q=100"""'
);

/* INSERT QUERY NO: 276 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_03', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/always-forever-wedding-invitations?v=9d977cc4-17aa-464d-a854-16bfef5eb335&q=100"""'
);

/* INSERT QUERY NO: 277 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_04', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/always-forever-wedding-invitations?v=9d977cc4-17aa-464d-a854-16bfef5eb335&q=100"""'
);

/* INSERT QUERY NO: 278 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_05', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/always-forever-wedding-invitations?v=9d977cc4-17aa-464d-a854-16bfef5eb335&q=100"""'
);

/* INSERT QUERY NO: 279 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_06', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""'
);

/* INSERT QUERY NO: 280 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_07', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""'
);

/* INSERT QUERY NO: 281 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_08', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""'
);

/* INSERT QUERY NO: 282 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_09', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""""'
);

/* INSERT QUERY NO: 283 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_10', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""'
);

/* INSERT QUERY NO: 284 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_11', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""""'
);

/* INSERT QUERY NO: 285 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_12', 'invitations', 'coffee n cream press', 'online', 0, 40, 0, '"""""https://pricing-sheets.s3.amazonaws.com/f94b58b1-fe1d-4310-bb46-8cd663bb33f8/5e1a138f67a644d039c0f82516e7d5e5604415cf/1d4eb8dc-b320-4884-b579-3ddd3444bd5f"""""'
);

/* INSERT QUERY NO: 286 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_13', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/cosmopolitan-wedding-response-cards?v=a65f360f-42a2-4aa0-bb65-ced0a84c71e9&q=100"""'
);

/* INSERT QUERY NO: 287 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_14', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/cosmopolitan-wedding-response-cards?v=a65f360f-42a2-4aa0-bb65-ced0a84c71e9&q=100"""'
);

/* INSERT QUERY NO: 288 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_15', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/gilded-monogram-place-cards?v=7e9f51c7-2f45-4fce-bbf2-6e8a81b9e0f9&q=100"""'
);

/* INSERT QUERY NO: 289 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_16', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/gilded-wildflower-table-numbers?v=4f047410-7ea9-4ba5-ae13-5b760c1191c8&q=100"""'
);

/* INSERT QUERY NO: 290 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_17', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/gilded-wildflower-table-numbers?v=4f047410-7ea9-4ba5-ae13-5b760c1191c8&q=100"""'
);

/* INSERT QUERY NO: 291 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_18', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/gilded-wreath-wedding-response-cards?v=402d9dcd-892d-43b7-8921-1d5dda653779&q=100"""'
);

/* INSERT QUERY NO: 292 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_19', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/gilded-wreath-wedding-response-cards?v=402d9dcd-892d-43b7-8921-1d5dda653779&q=100"""'
);

/* INSERT QUERY NO: 293 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_20', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/moonlit-garden-wedding-invitations?v=0857bc4a-2193-413a-b015-3b12c3343a7d&q=100"""'
);

/* INSERT QUERY NO: 294 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_21', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/moonlit-garden-wedding-invitations?v=0857bc4a-2193-413a-b015-3b12c3343a7d&q=100"""'
);

/* INSERT QUERY NO: 295 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_22', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/moonlit-garden-wedding-invitations?v=0857bc4a-2193-413a-b015-3b12c3343a7d&q=100"""'
);

/* INSERT QUERY NO: 296 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_23', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/moonlit-garden-wedding-invitations?v=0857bc4a-2193-413a-b015-3b12c3343a7d&q=100"""'
);

/* INSERT QUERY NO: 297 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_24', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/mountain-escape-wedding-response-cards?v=fd815e5e-ecc5-413f-adf1-85130b0a0dfd&q=100"""'
);

/* INSERT QUERY NO: 298 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_25', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/opulences-vera-wang-luxe-envelope-liners?v=ac102078-10c1-41cc-9e1e-c6e5a86f4a2d&q=100"""'
);

/* INSERT QUERY NO: 299 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_26', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/opulences-vera-wang-luxe-envelope-liners?v=ac102078-10c1-41cc-9e1e-c6e5a86f4a2d&q=100"""'
);

/* INSERT QUERY NO: 300 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_27', 'invitations', 'theknot', 'online', 0, 42, 0, '"""https://www.theknot.com/paper/product/opulences-vera-wang-luxe-envelope-liners?v=ac102078-10c1-41cc-9e1e-c6e5a86f4a2d&q=100"""'
);

/* INSERT QUERY NO: 301 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_28', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/opulences-vera-wang-programs?v=171fbd8b-2b9d-426a-9571-51d6e628520a&q=100"""'
);

/* INSERT QUERY NO: 302 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_29', 'invitations', 'etsy', 'online', 1, 45, 0, '"""https://www.etsy.com/listing/1366823112/plantable-envelopes-with-wildflower?gpla=1&gao=1&gad_source=1&variation0=3126922593"""'
);

/* INSERT QUERY NO: 303 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_30', 'invitations', 'etsy', 'online', 1, 45, 0, '"""https://www.etsy.com/listing/1366823112/plantable-envelopes-with-wildflower?gpla=1&gao=1&gad_source=1&variation0=3126922593"""'
);

/* INSERT QUERY NO: 304 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_31', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/rustic-forest-wedding-response-cards-p-23716.html#quantity=100&thickness=single&trim=round"""'
);

/* INSERT QUERY NO: 305 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_32', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/rustic-forest-wedding-response-cards-p-23716.html#quantity=100&thickness=single&trim=round"""'
);

/* INSERT QUERY NO: 306 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_33', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 307 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_34', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 308 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_35', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-table-cards-p-28127.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 309 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_36', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-programs-p-27902.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 310 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_37', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 311 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_38', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 312 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_39', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-menus-p-28125.html#quantity=50&thickness=single&trim=round"""'
);

/* INSERT QUERY NO: 313 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_40', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-response-cards-p-24970.html#quantity=100&thickness=triple&trim=straight"""'
);

/* INSERT QUERY NO: 314 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_41', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-programs-p-27902.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 315 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_42', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-envelope-liners-p-25236.html#quantity=50&thickness=rw"""'
);

/* INSERT QUERY NO: 316 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_43', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-envelope-liners-p-25236.html#quantity=50&thickness=rw"""'
);

/* INSERT QUERY NO: 317 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_44', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-name-cards-place-cards-p-28126.html#quantity=100&thickness=single"""'
);

/* INSERT QUERY NO: 318 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_45', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 319 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_46', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-invitation-suites-p-24969.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 320 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_47', 'invitations', 'paperculture', 'online', 1, 40, 0, '"""https://www.paperculture.com/eco/tahoe-inspired-wedding-table-cards-p-28127.html#quantity=100&thickness=single&trim=straight"""'
);

/* INSERT QUERY NO: 321 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_48', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/gs/dashboard"""'
);

/* INSERT QUERY NO: 322 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_49', 'invitations', 'theknot', 'online', 1, 42, 0, '"""https://www.theknot.com/paper/product/timeless-frame-menus?v=5cce5c47-22d9-4345-ab87-46d30af47b38&q=100"""'
);

/* INSERT QUERY NO: 323 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'inv_50', 'invitations', 'etsy', 'online', 1, 45, 0, '"""https://www.etsy.com/listing/1391871120/wedding-program-template-minimalist?ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=digital+wedding+program&ref=sr_gallery-1-1&pro=1&sts=1&dd=1&organic_search_click=1"""'
);
```
### Inserting Values (Music Department)
```
/* INSERT QUERY NO: 324 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_01', 'music', 'the celebration dj', 'oakland ', 0, 49, 3039310812, ''
);

/* INSERT QUERY NO: 325 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_02', 'music', 'all ears', 'san francisco ', 1, 49, 7857662596, ''
);

/* INSERT QUERY NO: 326 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_03', 'music', 'big carli llc', 'oakland ', 0, 50, 5102196323, ''
);

/* INSERT QUERY NO: 327 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_04', 'music', 'kyanni productions', 'santa clara ', 0, 50, 5107554086, ''
);

/* INSERT QUERY NO: 328 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_05', 'music', 'sounds elevated', 'berkeley ', 0, 50, 9705966942, ''
);

/* INSERT QUERY NO: 329 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_06', 'music', 'dj alex reyes entertainment', 'freemont ', 1, 50, 5107545383, ''
);

/* INSERT QUERY NO: 330 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_07', 'music', 'dj buddy holly', 'san mateo ', 1, 45, 6503391131, ''
);

/* INSERT QUERY NO: 331 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_08', 'music', 'heart of gold ', ' san francisco ', 1, 45, 4159350744, ''
);

/* INSERT QUERY NO: 332 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_09', 'music', 'sf deejays', 'san francisco ', 1, 50, 4158761350, ''
);

/* INSERT QUERY NO: 333 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_10', 'music', 'total dj', 'san francisco ', 0, 50, 4157411535, ''
);

/* INSERT QUERY NO: 334 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_11', 'music', 'dj by the bay', 'san francisco ', 0, 50, 5102057802, ''
);

/* INSERT QUERY NO: 335 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_12', 'music', 'imobile djs', 'hayward ', 0, 50, 8334635462, ''
);

/* INSERT QUERY NO: 336 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_13', 'music', 'all soiree dj', 'san jose ', 1, 50, 4084172102, ''
);

/* INSERT QUERY NO: 337 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_14', 'music', 'ds entertainment ', ' concord ', 1, 50, 9252021090, ''
);

/* INSERT QUERY NO: 338 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_15', 'music', 'oui boogie', 'san mateo ', 1, 48, 5105574272, ''
);

/* INSERT QUERY NO: 339 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_16', 'music', 'plural music', 'oakland ', 0, 48, 2099867431, ''
);

/* INSERT QUERY NO: 340 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_17', 'music', 'big fun disc jockeys', 'cupertino ', 1, 43, 8009244386, ''
);

/* INSERT QUERY NO: 341 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_18', 'music', 'fantasy sound event services', 'livermore ', 0, 49, 9253731934, ''
);

/* INSERT QUERY NO: 342 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_19', 'music', 'dj laozyb', 'san jose ', 1, 50, 6692633205, ''
);

/* INSERT QUERY NO: 343 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_20', 'music', 'dj johny', 'san jose ', 0, 50, 4086584568, ''
);

/* INSERT QUERY NO: 344 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_21', 'music', 'one way music', 'san jose ', 1, 30, 4088919251, ''
);

/* INSERT QUERY NO: 345 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_22', 'music', 'sounds spin n dj', 'san mateo ', 1, 50, 6505180543, ''
);

/* INSERT QUERY NO: 346 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_23', 'music', 'local productions mobile dj', 'san mateo ', 1, 50, 6289459467, ''
);

/* INSERT QUERY NO: 347 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_24', 'music', 'majestick events', 'freemont ', 1, 49, 5109950688, ''
);

/* INSERT QUERY NO: 348 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_25', 'music', 'music plus events', 'freemont ', 1, 48, 5107708787, ''
);

/* INSERT QUERY NO: 349 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_26', 'music', 'silver sound productions', 'concord ', 1, 50, 4154659445, ''
);

/* INSERT QUERY NO: 350 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_27', 'music', 'premier dj service', 'concord ', 1, 47, 9253789367, ''
);

/* INSERT QUERY NO: 351 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_28', 'music', 'dj new', 'mountain view ', 0, 50, 4082428250, ''
);

/* INSERT QUERY NO: 352 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_29', 'music', 'shining city music ent', 'mountain view ', 0, 49, 8447744646, ''
);

/* INSERT QUERY NO: 353 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_30', 'music', 'dj sneak 415', 'novato ', 0, 49, 4152502194, ''
);

/* INSERT QUERY NO: 354 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_31', 'music', 'grand slam mobile djs', 'novato ', 1, 50, 4158979270, ''
);

/* INSERT QUERY NO: 355 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_32', 'music', 'tde wedding events', 'milbrae ', 1, 50, 4158138900, ''
);

/* INSERT QUERY NO: 356 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_33', 'music', 'km audiovisual', 'brentwood ', 0, 0, 8002750772, ''
);

/* INSERT QUERY NO: 357 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_34', 'music', 'sound wave mobile dj', 'dublin ', 0, 0, 9252891800, ''
);

/* INSERT QUERY NO: 358 */
INSERT INTO Vendors(vendor_id, vendor_depart, vendor_name, vendor_location, vendor_sustainable, vendor_rating, vendor_contact, vendor_website)
VALUES
(
'dj_35', 'music', 'los gatos dj company', 'los gatos ', 1, 0, 4083567717, ''
);
```
### V. Wedding Products
### Venue Department
```
/* INSERT QUERY NO: 1 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_001', 'the bridges golf club', 'ven_01', 14000, '1 per vendor', 2
);

/* INSERT QUERY NO: 2 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_002', 'viaggio estate & winery', 'ven_02', 9500, '1 per vendor', 1
);

/* INSERT QUERY NO: 3 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_003', 'leal vineyards', 'ven_03', 27000, '1 per vendor', 2
);

/* INSERT QUERY NO: 4 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_004', 'oceano hotel & spa', 'ven_04', 16000, '1 per vendor', 2
);

/* INSERT QUERY NO: 5 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_005', 'log cabin at the presidio', 'ven_05', 24000, '1 per vendor', 2
);

/* INSERT QUERY NO: 6 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_006', 'callippe preserve', 'ven_06', 10000, '1 per vendor', 2
);

/* INSERT QUERY NO: 7 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_007', 'the garden walnut creek', 'ven_07', 18500, '1 per vendor', 3
);

/* INSERT QUERY NO: 8 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_008', 'the terrace room at lake merritt', 'ven_08', 15000, '1 per vendor', 2
);

/* INSERT QUERY NO: 9 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_009', 'berkeley country club', 'ven_09', 9000, '1 per vendor', 2
);

/* INSERT QUERY NO: 10 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_010', 'pinstripes san mateo', 'ven_10', 4500, '1 per vendor', 2
);

/* INSERT QUERY NO: 11 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_011', 'the club at castlewood', 'ven_11', 6000, '1 per vendor', 2
);

/* INSERT QUERY NO: 12 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_012', 'berkeley city club', 'ven_12', 24000, '1 per vendor', 3
);

/* INSERT QUERY NO: 13 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_013', 'marines memorial club and hotel union square', 'ven_13', 4750, '1 per vendor', 2
);

/* INSERT QUERY NO: 14 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_014', 'rancho nicasio', 'ven_14', 20000, '1 per vendor', 2
);

/* INSERT QUERY NO: 15 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_015', 'fairview metropolitan', 'ven_15', 10000, '1 per vendor', 2
);

/* INSERT QUERY NO: 16 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_016', 'corinthian yacht club', 'ven_16', 20000, '1 per vendor', 3
);

/* INSERT QUERY NO: 17 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_017', 'coyote creek golf club', 'ven_17', 3000, '1 per vendor', 2
);

/* INSERT QUERY NO: 18 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_018', 'the purple orchid wine country resort & spa', 'ven_18', 22000, '1 per vendor', 3
);

/* INSERT QUERY NO: 19 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_019', 'yin ranch', 'ven_19', 16500, '1 per vendor', 3
);

/* INSERT QUERY NO: 20 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_020', 'the university club of san francisco', 'ven_20', 20000, '1 per vendor', 2
);

/* INSERT QUERY NO: 21 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_021', 'san jose marriott', 'ven_21', 14000, '1 per vendor', 3
);

/* INSERT QUERY NO: 22 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_022', 'beacon grand', 'ven_22', 18000, '1 per vendor', 3
);

/* INSERT QUERY NO: 23 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_023', 'oakhurst county club', 'ven_23', 2000, '1 per vendor', 3
);

/* INSERT QUERY NO: 24 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_024', 'hotel nia autograph collection', 'ven_24', 30000, '1 per vendor', 3
);

/* INSERT QUERY NO: 25 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_025', 'mammoth mountain ski area', 'ven_25', 21000, '1 per vendor', 3
);

/* INSERT QUERY NO: 26 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_026', 'mountain house estate', 'ven_26', 20000, '1 per vendor', 2
);

/* INSERT QUERY NO: 27 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_027', 'roaring camp railroads', 'ven_27', 9000, '1 per vendor', 2
);

/* INSERT QUERY NO: 28 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_028', 'curiodyssey', 'ven_28', 5000, '1 per vendor', 1
);

/* INSERT QUERY NO: 29 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_029', 'babette', 'ven_29', 8000, '1 per vendor', 2
);

/* INSERT QUERY NO: 30 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_030', 'stanford faculty club', 'ven_30', 10370, '1 per vendor', 2
);

/* INSERT QUERY NO: 31 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_031', 'the monk ranch', 'ven_31', 4700, '1 per vendor', 2
);

/* INSERT QUERY NO: 32 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_032', 'green oaks creek farm', 'ven_32', 19200, '1 per vendor', 2
);

/* INSERT QUERY NO: 33 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_033', 'hilton san jose', 'ven_33', 4000, '1 per vendor', 2
);

/* INSERT QUERY NO: 34 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_034', 'university of redlands marin campus', 'ven_34', 4800, '1 per vendor', 2
);

/* INSERT QUERY NO: 35 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_035', 'grand hyatt at sfo', 'ven_35', 10000, '1 per vendor', 2
);

/* INSERT QUERY NO: 36 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_036', 'unity in marin weddings', 'ven_36', 10000, '1 per vendor', 2
);

/* INSERT QUERY NO: 37 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_037', 'regale winery and vineyards', 'ven_37', 24000, '1 per vendor', 3
);

/* INSERT QUERY NO: 38 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_038', 'swedish american hall', 'ven_38', 7500, '1 per vendor', 3
);

/* INSERT QUERY NO: 39 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_039', 'elliston vineyards', 'ven_39', 5100, '1 per vendor', 3
);

/* INSERT QUERY NO: 40 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_040', 'solage calistoga', 'ven_40', 32000, '1 per vendor', 3
);

/* INSERT QUERY NO: 41 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_041', 'riggers loft wine company', 'ven_41', 8000, '1 per vendor', 3
);

/* INSERT QUERY NO: 42 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_042', 'blanco urban venue', 'ven_42', 16980, '1 per vendor', 2
);

/* INSERT QUERY NO: 43 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_043', 'brownstone gardens', 'ven_43', 4500, '1 per vendor', 3
);

/* INSERT QUERY NO: 44 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_044', 'villa ragusa', 'ven_44', 2450, '1 per vendor', 3
);

/* INSERT QUERY NO: 45 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_045', 'sf museum of modern art', 'ven_45', 32000, '1 per vendor', 4
);

/* INSERT QUERY NO: 46 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_046', 'cavallo point the lodge at the golden gate', 'ven_46', 9500, '1 per vendor', 3
);

/* INSERT QUERY NO: 47 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_047', 'webster hall sf', 'ven_47', 5000, '1 per vendor', 2
);

/* INSERT QUERY NO: 48 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_048', 'the pearl', 'ven_48', 30000, '1 per vendor', 4
);

/* INSERT QUERY NO: 49 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_049', 'half moon bay golf links', 'ven_49', 10000, '1 per vendor', 3
);

/* INSERT QUERY NO: 50 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_050', 'omni san francisco hotel', 'ven_50', 5000, '1 per vendor', 3
);
```
### Rental Department
```
/* INSERT QUERY NO: 51 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_051', 'linen ', 'ren_01', 4, '1 per table', 3
);

/* INSERT QUERY NO: 52 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_052', 'linen', 'ren_13', 3, '1 per table', 2
);

/* INSERT QUERY NO: 53 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_053', 'linen', 'ren_14', 3, '1 per table', 2
);

/* INSERT QUERY NO: 54 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_054', 'linen', 'ren_16', 1.5, '1 per table', 2
);

/* INSERT QUERY NO: 55 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_055', 'linen', 'ren_23', 0, '1 per table', 0
);

/* INSERT QUERY NO: 56 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_056', 'linen', 'ren_11', 0, '1 per table', 0
);

/* INSERT QUERY NO: 57 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_057', 'backdrops', 'ren_05', 750, '1 per backdrop', 4
);

/* INSERT QUERY NO: 58 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_058', 'backdrops', 'ren_06', 800, '1 per backdrop', 4
);

/* INSERT QUERY NO: 59 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_059', 'backdrops', 'ren_18', 750, '1 per backdrop', 4
);

/* INSERT QUERY NO: 60 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_060', 'backdrops', 'ren_19', 500, '1 per backdrop', 4
);

/* INSERT QUERY NO: 61 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_061', 'dinnerware', 'ren_04', 0.52, '1 per piece', 1
);

/* INSERT QUERY NO: 62 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_062', 'dinnerware', 'ren_08', 0.85, '1 per piece', 1
);

/* INSERT QUERY NO: 63 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_063', 'dinnerware', 'ren_22', 1, '1 per piece', 2
);

/* INSERT QUERY NO: 64 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_064', 'dinnerware', 'ren_13', 0.7, '1 per piece', 1
);

/* INSERT QUERY NO: 65 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_065', 'dinnerware', 'ren_10', 0.7, '1 per piece', 1
);

/* INSERT QUERY NO: 66 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_066', 'glasswares', 'ren_17', 0.7, '1 per piece', 1
);

/* INSERT QUERY NO: 67 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_067', 'glasswares', 'ren_02', 0.8, '1 per piece', 1
);

/* INSERT QUERY NO: 68 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_068', 'glasswares', 'ren_07', 2, '1 per piece', 2
);

/* INSERT QUERY NO: 69 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_069', 'glasswares', 'ren_12', 0.73, '1 per piece', 1
);

/* INSERT QUERY NO: 70 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_070', 'glasswares', 'ren_21', 0.8, '1 per piece', 1
);

/* INSERT QUERY NO: 71 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_071', 'tables', 'ren_21', 11.5, '1 per table', 3
);

/* INSERT QUERY NO: 72 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_072', 'tables', 'ren_12', 9.25, '1 per table', 3
);

/* INSERT QUERY NO: 73 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_073', 'tables', 'ren_03', 9.25, '1 per table', 3
);

/* INSERT QUERY NO: 74 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_074', 'tables', 'ren_24', 25, '1 per table', 4
);

/* INSERT QUERY NO: 75 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_075', 'tables', 'ren_25', 0, '1 per table', 0
);

/* INSERT QUERY NO: 76 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_076', 'tables', 'ren_20', 12, '1 per table', 4
);

/* INSERT QUERY NO: 77 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_077', 'tables', 'ren_11', 9, '1 per table', 3
);

/* INSERT QUERY NO: 78 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_078', 'tables', 'ren_15', 75, '1 per table', 4
);

/* INSERT QUERY NO: 79 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_079', 'chairs', 'ren_10', 18, '1 per chair', 4
);

/* INSERT QUERY NO: 80 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_080', 'chairs', 'ren_11', 1.5, '1 per chair', 2
);

/* INSERT QUERY NO: 81 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_081', 'chairs', 'ren_16', 1.5, '1 per chair', 2
);

/* INSERT QUERY NO: 82 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_082', 'chairs', 'ren_09', 8, '1 per chair', 3
);

/* INSERT QUERY NO: 83 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_083', 'chairs', 'ren_21', 4.5, '1 per chair', 3
);
```
### Jewelry Department
```
/* INSERT QUERY NO: 84 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_084', 'sol-starburst-diamond', 'jwl_01', 1390, '1', 2
);

/* INSERT QUERY NO: 85 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_085', 'lunette-diamond-ring', 'jwl_01', 1250, '1', 2
);

/* INSERT QUERY NO: 86 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_086', 'lola-paperclip-necklace', 'jwl_01', 650, '1', 1
);

/* INSERT QUERY NO: 87 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_087', 'sol starburst diamond', 'jwl_01', 2995, '1', 3
);

/* INSERT QUERY NO: 88 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_088', 'lola-paperclip-necklace', 'jwl_01', 450, '1', 1
);

/* INSERT QUERY NO: 89 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_089', 'trio-diamond-pendant', 'jwl_01', 1190, '1', 2
);

/* INSERT QUERY NO: 90 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_090', 'milo-24-in.-rope-chain-necklace', 'jwl_01', 995, '1', 1
);

/* INSERT QUERY NO: 91 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_091', 'whisper 6.5 in. diamond bangle bracelet', 'jwl_01', 1795, '1', 2
);

/* INSERT QUERY NO: 92 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_092', 'hydrangea bouquet cuff bracelet', 'jwl_01', 295, '1', 1
);

/* INSERT QUERY NO: 93 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_093', 'jardiniere-diamond-ring', 'jwl_01', 1790, '1', 2
);

/* INSERT QUERY NO: 94 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_094', 'athena premium akoya cultured pearl bracelet', 'jwl_01', 590, '1', 1
);

/* INSERT QUERY NO: 95 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_095', 'dezi-18-in.-herringbone-necklace', 'jwl_01', 750, '1', 1
);

/* INSERT QUERY NO: 96 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_096', 'caris-diamond-ring', 'jwl_01', 1190, '1', 2
);

/* INSERT QUERY NO: 97 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_097', 'emi bracelet', 'jwl_02', 3025, '1', 3
);

/* INSERT QUERY NO: 98 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_098', 'solitaire claw prong + french pavé eternity band ', 'jwl_02', 4480, '1', 4
);

/* INSERT QUERY NO: 99 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_099', 'riley ring ', 'jwl_02', 210, '1', 1
);

/* INSERT QUERY NO: 100 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_100', 'gwendolyn bracelets', 'jwl_02', 200, '1', 1
);

/* INSERT QUERY NO: 101 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_101', 'jules ring ', 'jwl_02', 2575, '1', 3
);

/* INSERT QUERY NO: 102 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_102', 'imogen ring ', 'jwl_02', 1935, '1', 3
);

/* INSERT QUERY NO: 103 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_103', 'dove ring ', 'jwl_02', 260, '1', 1
);

/* INSERT QUERY NO: 104 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_104', 'valencia ring ', 'jwl_02', 6700, '1', 4
);

/* INSERT QUERY NO: 105 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_105', 'hugo bracelet', 'jwl_02', 1375, '1', 2
);

/* INSERT QUERY NO: 106 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_106', 'tiffany ring ', 'jwl_02', 1575, '1', 2
);

/* INSERT QUERY NO: 107 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_107', 'calista ring ', 'jwl_02', 2175, '1', 3
);

/* INSERT QUERY NO: 108 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_108', 'daphne ring ', 'jwl_02', 1120, '1', 2
);

/* INSERT QUERY NO: 109 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_109', 'ella necklace', 'jwl_02', 1375, '1', 2
);

/* INSERT QUERY NO: 110 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_110', 'halo engagement ring', 'jwl_03', 3550, '1', 4
);

/* INSERT QUERY NO: 111 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_111', 'dot diamonds engagement ring', 'jwl_03', 2750, '1', 3
);

/* INSERT QUERY NO: 112 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_112', '"alex sepkus platinum sapphire and diamond ""candy"" dome ring"', 'jwl_03', 4295, '1', 4
);

/* INSERT QUERY NO: 113 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_113', 'forevermark round halo 5 stone diamond ring', 'jwl_03', 5890, '1', 4
);

/* INSERT QUERY NO: 114 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_114', 'diamond engagement ring', 'jwl_03', 3150, '1', 3
);

/* INSERT QUERY NO: 115 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_115', 'oval diamond engagement ring', 'jwl_03', 2850, '1', 3
);

/* INSERT QUERY NO: 116 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_116', 'diamond tennis bracelet', 'jwl_03', 5185, '1', 4
);

/* INSERT QUERY NO: 117 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_117', 'diamond bracelet', 'jwl_03', 750, '1', 1
);

/* INSERT QUERY NO: 118 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_118', 'diamond tennis bracelet', 'jwl_03', 1895, '1', 2
);

/* INSERT QUERY NO: 119 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_119', 'diamond bracelet', 'jwl_03', 4975, '1', 4
);

/* INSERT QUERY NO: 120 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_120', 'diamond bracelet', 'jwl_03', 4395, '1', 4
);

/* INSERT QUERY NO: 121 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_121', 'diamond necklace', 'jwl_03', 3750, '1', 4
);

/* INSERT QUERY NO: 122 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_122', 'beleza necklace', 'jwl_04', 2308, '1', 3
);

/* INSERT QUERY NO: 123 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_123', 'valencia necklace', 'jwl_04', 2319, '1', 3
);

/* INSERT QUERY NO: 124 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_124', 'vitus necklace', 'jwl_04', 832, '1', 1
);

/* INSERT QUERY NO: 125 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_125', 'solvus necklace', 'jwl_04', 930, '1', 1
);

/* INSERT QUERY NO: 126 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_126', 'ewan ring', 'jwl_04', 1523, '1', 2
);

/* INSERT QUERY NO: 127 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_127', 'elqan ring', 'jwl_04', 4814, '1', 4
);

/* INSERT QUERY NO: 128 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_128', 'volary ring', 'jwl_04', 1275, '1', 2
);

/* INSERT QUERY NO: 129 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_129', 'astuty ring', 'jwl_04', 1940, '1', 3
);

/* INSERT QUERY NO: 130 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_130', 'conway bracelet', 'jwl_04', 3516, '1', 4
);

/* INSERT QUERY NO: 131 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_131', 'calida bracelet', 'jwl_04', 5588, '1', 4
);

/* INSERT QUERY NO: 132 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_132', 'helen bracelet', 'jwl_04', 2321, '1', 3
);

/* INSERT QUERY NO: 133 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_133', 'nicole bracelet', 'jwl_04', 2037, '1', 3
);
```
### Invitation Department
```
/* INSERT QUERY NO: 134 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_134', 'invitationalways and forever', 'inv_01', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 135 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_135', 'invitationalways and forever', 'inv_02', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 136 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_136', 'invitationalways and forever', 'inv_03', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 137 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_137', 'invitationalways and forever', 'inv_04', 324.96, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 138 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_138', 'invitationalways and forever', 'inv_05', 324.96, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 139 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_139', 'envelope linercoffee n cream press', 'inv_06', 88.8, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 140 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_140', 'envelope linercoffee n cream press', 'inv_07', 88.8, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 141 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_141', 'envelope linercoffee n cream press', 'inv_08', 88.8, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 142 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_142', 'invitationcoffee n cream press', 'inv_09', 297.88, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 143 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_143', 'invitationcoffee n cream press', 'inv_10', 297.88, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 144 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_144', 'rsvpcoffee n cream press', 'inv_11', 284.34, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 145 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_145', 'rsvpcoffee n cream press', 'inv_12', 284.34, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 146 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_146', 'rsvpcosmopolitan', 'inv_13', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 147 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_147', 'rsvpcosmopolitan', 'inv_14', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 148 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_148', 'place cardgilded monogram', 'inv_15', 352.04, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 149 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_149', 'table numbergilded wildflower', 'inv_16', 352.04, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 150 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_150', 'table numbergilded wildflower', 'inv_17', 324.96, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 151 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_151', 'rsvpgilded wreath', 'inv_18', 284.34, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 152 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_152', 'rsvpgilded wreath', 'inv_19', 284.34, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 153 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_153', 'invitationmoonlit garden', 'inv_20', 284.34, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 154 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_154', 'invitationmoonlit garden', 'inv_21', 284.34, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 155 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_155', 'invitationmoonlit garden', 'inv_22', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 156 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_156', 'invitationmoonlit garden', 'inv_23', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 157 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_157', 'rsvpmountain escape', 'inv_24', 324.96, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 158 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_158', 'envelope lineropulences by vera wang', 'inv_25', 88.8, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 159 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_159', 'envelope lineropulences by vera wang', 'inv_26', 88.8, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 160 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_160', 'envelope lineropulences by vera wang', 'inv_27', 88.8, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 161 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_161', 'wedding programopulences by vera wang', 'inv_28', 352.04, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 162 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_162', 'envelope linerrecycled and plantable envelope', 'inv_29', 125.4, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 163 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_163', 'envelope linerrecycled and plantable envelope', 'inv_30', 125.4, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 164 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_164', 'rsvprustic forest', 'inv_31', 392.66, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 165 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_165', 'rsvprustic forest', 'inv_32', 392.66, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 166 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_166', 'invitationtahoe inspired', 'inv_33', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 167 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_167', 'invitationtahoe inspired', 'inv_34', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 168 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_168', 'table numbertahoe inspired', 'inv_35', 392.66, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 169 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_169', 'wedding programtahoe inspired', 'inv_36', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 170 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_170', 'invitationtahoe inspired', 'inv_37', 392.66, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 171 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_171', 'invitationtahoe inspired', 'inv_38', 392.66, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 172 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_172', 'menutahoe inspired', 'inv_39', 392.66, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 173 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_173', 'rsvptahoe inspired', 'inv_40', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 174 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_174', 'wedding programtahoe inspired', 'inv_41', 392.66, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 175 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_175', 'envelope linertahoe inspired', 'inv_42', 118.4, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 176 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_176', 'envelope linertahoe inspired', 'inv_43', 118.4, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 177 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_177', 'place cardtahoe inspired', 'inv_44', 352.04, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 178 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_178', 'invitationtahoe inspired', 'inv_45', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 179 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_179', 'invitationtahoe inspired', 'inv_46', 392.66, 'per 100 invitations', 2
);

/* INSERT QUERY NO: 180 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_180', 'table numbertahoe inspired', 'inv_47', 392.66, 'per 100 invitations', 4
);

/* INSERT QUERY NO: 181 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_181', 'rsvpthe knot wedding website', 'inv_48', 50, 'per 100 invitations', 1
);

/* INSERT QUERY NO: 182 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_182', 'menutimeless frame', 'inv_49', 352.04, 'per 100 invitations', 3
);

/* INSERT QUERY NO: 183 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_183', 'wedding programdigital', 'inv_50', 50, 'per 100 invitations', 4
);
```
### Flowers Department
```
/* INSERT QUERY NO: 184 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_184', 'corsage', 'flo_01', 30, 'unit', 4
);

/* INSERT QUERY NO: 185 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_185', 'corsage', 'flo_02', 15, 'unit', 1
);

/* INSERT QUERY NO: 186 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_186', 'bouquet', 'flo_03', 175, 'unit', 2
);

/* INSERT QUERY NO: 187 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_187', 'flower petals', 'flo_04', 3, 'unit', 1
);

/* INSERT QUERY NO: 188 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_188', 'flower petals', 'flo_05', 5, 'unit', 1
);

/* INSERT QUERY NO: 189 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_189', 'bouquet', 'flo_06', 350, 'unit', 4
);

/* INSERT QUERY NO: 190 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_190', 'flower petals', 'flo_07', 17, 'unit', 3
);

/* INSERT QUERY NO: 191 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_191', 'boutounneries', 'flo_08', 15, 'unit', 2
);

/* INSERT QUERY NO: 192 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_192', 'corsage', 'flo_09', 25, 'unit', 3
);

/* INSERT QUERY NO: 193 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_193', 'single flower', 'flo_10', 7.5, 'unit', 1
);

/* INSERT QUERY NO: 194 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_194', 'corsage', 'flo_11', 15, 'unit', 1
);

/* INSERT QUERY NO: 195 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_195', 'flower crowns', 'flo_12', 45, 'unit', 2
);

/* INSERT QUERY NO: 196 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_196', 'corsage', 'flo_13', 20, 'unit', 2
);

/* INSERT QUERY NO: 197 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_197', 'flowers arrangement', 'flo_14', 65, 'unit', 1
);

/* INSERT QUERY NO: 198 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_198', 'bouquet', 'flo_15', 300, 'unit', 4
);

/* INSERT QUERY NO: 199 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_199', 'single flower', 'flo_16', 12, 'unit', 2
);

/* INSERT QUERY NO: 200 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_200', 'flower petals', 'flo_17', 8, 'unit', 2
);

/* INSERT QUERY NO: 201 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_201', 'flower crowns', 'flo_18', 60, 'unit', 4
);

/* INSERT QUERY NO: 202 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_202', 'flower crowns', 'flo_19', 35, 'unit', 1
);

/* INSERT QUERY NO: 203 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_203', 'flowers arrangement', 'flo_20', 150, 'unit', 3
);

/* INSERT QUERY NO: 204 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_204', 'boutounneries', 'flo_21', 15, 'unit', 2
);

/* INSERT QUERY NO: 205 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_205', 'flowers arrangement', 'flo_22', 130, 'unit', 2
);

/* INSERT QUERY NO: 206 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_206', 'bouquet', 'flo_23', 80, 'unit', 1
);

/* INSERT QUERY NO: 207 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_207', 'flower crowns', 'flo_24', 50, 'unit', 3
);

/* INSERT QUERY NO: 208 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_208', 'flower crowns', 'flo_25', 60, 'unit', 4
);

/* INSERT QUERY NO: 209 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_209', 'flower petals', 'flo_26', 12, 'unit', 2
);

/* INSERT QUERY NO: 210 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_210', 'flowers arrangement', 'flo_27', 65, 'unit', 1
);

/* INSERT QUERY NO: 211 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_211', 'boutounneries', 'flo_28', 20, 'unit', 3
);

/* INSERT QUERY NO: 212 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_212', 'single flower', 'flo_29', 20, 'unit', 4
);

/* INSERT QUERY NO: 213 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_213', 'flower petals', 'flo_30', 10, 'unit', 2
);

/* INSERT QUERY NO: 214 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_214', 'flowers arrangement', 'flo_31', 100, 'unit', 2
);

/* INSERT QUERY NO: 215 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_215', 'bouquet', 'flo_32', 250, 'unit', 3
);

/* INSERT QUERY NO: 216 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_216', 'corsage', 'flo_33', 25, 'unit', 3
);

/* INSERT QUERY NO: 217 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_217', 'bouquet', 'flo_34', 185, 'unit', 2
);

/* INSERT QUERY NO: 218 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_218', 'corsage', 'flo_35', 20, 'unit', 2
);

/* INSERT QUERY NO: 219 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_219', 'single flower', 'flo_36', 16, 'unit', 3
);

/* INSERT QUERY NO: 220 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_220', 'boutounneries', 'flo_37', 22, 'unit', 4
);

/* INSERT QUERY NO: 221 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_221', 'flower petals', 'flo_38', 20, 'unit', 4
);

/* INSERT QUERY NO: 222 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_222', 'boutounneries', 'flo_39', 22, 'unit', 4
);

/* INSERT QUERY NO: 223 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_223', 'bouquet', 'flo_40', 210, 'unit', 3
);

/* INSERT QUERY NO: 224 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_224', 'flowers arrangement', 'flo_41', 200, 'unit', 4
);

/* INSERT QUERY NO: 225 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_225', 'single flower', 'flo_42', 15, 'unit', 3
);
```
### Catering Department
```
/* INSERT QUERY NO: 226 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_226', 'plated service', 'cat_08', 100, 'per person', 2
);

/* INSERT QUERY NO: 227 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_227', 'plated service', 'cat_10', 72, 'per person', 2
);

/* INSERT QUERY NO: 228 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_228', 'plated service', 'cat_16', 35, 'per person', 2
);

/* INSERT QUERY NO: 229 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_229', 'plated service', 'cat_20', 60, 'per person', 3
);

/* INSERT QUERY NO: 230 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_230', 'plated service', 'cat_21', 125, 'per person', 3
);

/* INSERT QUERY NO: 231 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_231', 'plated service', 'cat_22', 155, 'per person', 3
);

/* INSERT QUERY NO: 232 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_232', 'plated service', 'cat_23', 110, 'per person', 3
);

/* INSERT QUERY NO: 233 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_233', 'plated service', 'cat_24', 136, 'per person', 3
);

/* INSERT QUERY NO: 234 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_234', 'plated service', 'cat_25', 150, 'per person', 3
);

/* INSERT QUERY NO: 235 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_235', 'plated service', 'cat_27', 150, 'per person', 3
);

/* INSERT QUERY NO: 236 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_236', 'plated service', 'cat_28', 8, 'per person', 3
);

/* INSERT QUERY NO: 237 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_237', 'plated service', 'cat_29', 175, 'per person', 3
);

/* INSERT QUERY NO: 238 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_238', 'plated service', 'cat_35', 75, 'per person', 3
);

/* INSERT QUERY NO: 239 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_239', 'plated service', 'cat_36', 120, 'per person', 3
);

/* INSERT QUERY NO: 240 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_240', 'plated service', 'cat_39', 250, 'per person', 3
);

/* INSERT QUERY NO: 241 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_241', 'plated service', 'cat_40', 20, 'per person', 3
);

/* INSERT QUERY NO: 242 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_242', 'plated service', 'cat_41', 85, 'per person', 3
);

/* INSERT QUERY NO: 243 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_243', 'plated service', 'cat_43', 130, 'per person', 3
);

/* INSERT QUERY NO: 244 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_244', 'plated service', 'cat_45', 180, 'per person', 3
);

/* INSERT QUERY NO: 245 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_245', 'plated service', 'cat_46', 130, 'per person', 3
);

/* INSERT QUERY NO: 246 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_246', 'plated service', 'cat_48', 145, 'per person', 4
);

/* INSERT QUERY NO: 247 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_247', 'plated service', 'cat_50', 250, 'per person', 4
);

/* INSERT QUERY NO: 248 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_248', 'buffet service', 'cat_08', 60, 'per person', 2
);

/* INSERT QUERY NO: 249 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_249', 'buffet service', 'cat_10', 64, 'per person', 2
);

/* INSERT QUERY NO: 250 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_250', 'buffet service', 'cat_16', 29, 'per person', 2
);

/* INSERT QUERY NO: 251 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_251', 'buffet service', 'cat_20', 55, 'per person', 3
);

/* INSERT QUERY NO: 252 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_252', 'buffet service', 'cat_21', 110, 'per person', 3
);

/* INSERT QUERY NO: 253 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_253', 'buffet service', 'cat_22', 95, 'per person', 3
);

/* INSERT QUERY NO: 254 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_254', 'buffet service', 'cat_23', 90, 'per person', 3
);

/* INSERT QUERY NO: 255 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_255', 'buffet service', 'cat_24', 85, 'per person', 3
);

/* INSERT QUERY NO: 256 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_256', 'buffet service', 'cat_25', 125, 'per person', 3
);

/* INSERT QUERY NO: 257 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_257', 'buffet service', 'cat_27', 150, 'per person', 3
);

/* INSERT QUERY NO: 258 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_258', 'buffet service', 'cat_28', 8, 'per person', 3
);

/* INSERT QUERY NO: 259 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_259', 'buffet service', 'cat_29', 80, 'per person', 3
);

/* INSERT QUERY NO: 260 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_260', 'buffet service', 'cat_35', 38, 'per person', 3
);

/* INSERT QUERY NO: 261 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_261', 'buffet service', 'cat_36', 90, 'per person', 3
);

/* INSERT QUERY NO: 262 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_262', 'buffet service', 'cat_39', 175, 'per person', 3
);

/* INSERT QUERY NO: 263 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_263', 'buffet service', 'cat_40', 190, 'per person', 3
);

/* INSERT QUERY NO: 264 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_264', 'buffet service', 'cat_41', 70, 'per person', 3
);

/* INSERT QUERY NO: 265 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_265', 'buffet service', 'cat_43', 75, 'per person', 3
);

/* INSERT QUERY NO: 266 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_266', 'buffet service', 'cat_45', 175, 'per person', 3
);

/* INSERT QUERY NO: 267 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_267', 'buffet service', 'cat_46', 120, 'per person', 3
);

/* INSERT QUERY NO: 268 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_268', 'buffet service', 'cat_50', 200, 'per person', 4
);

/* INSERT QUERY NO: 269 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_269', 'food stations', 'cat_08', 65, 'per person', 2
);

/* INSERT QUERY NO: 270 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_270', 'food stations', 'cat_10', 62, 'per person', 2
);

/* INSERT QUERY NO: 271 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_271', 'food stations', 'cat_16', 35, 'per person', 2
);

/* INSERT QUERY NO: 272 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_272', 'food stations', 'cat_20', 55, 'per person', 3
);

/* INSERT QUERY NO: 273 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_273', 'food stations', 'cat_21', 125, 'per person', 3
);

/* INSERT QUERY NO: 274 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_274', 'food stations', 'cat_22', 65, 'per person', 3
);

/* INSERT QUERY NO: 275 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_275', 'food stations', 'cat_23', 90, 'per person', 3
);

/* INSERT QUERY NO: 276 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_276', 'food stations', 'cat_24', 85, 'per person', 3
);

/* INSERT QUERY NO: 277 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_277', 'food stations', 'cat_25', 125, 'per person', 3
);

/* INSERT QUERY NO: 278 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_278', 'food stations', 'cat_27', 150, 'per person', 3
);

/* INSERT QUERY NO: 279 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_279', 'food stations', 'cat_28', 8, 'per person', 3
);

/* INSERT QUERY NO: 280 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_280', 'food stations', 'cat_29', 125, 'per person', 3
);

/* INSERT QUERY NO: 281 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_281', 'food stations', 'cat_35', 64, 'per person', 3
);

/* INSERT QUERY NO: 282 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_282', 'food stations', 'cat_36', 95, 'per person', 3
);

/* INSERT QUERY NO: 283 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_283', 'food stations', 'cat_40', 45, 'per person', 3
);

/* INSERT QUERY NO: 284 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_284', 'food stations', 'cat_41', 70, 'per person', 3
);

/* INSERT QUERY NO: 285 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_285', 'food stations', 'cat_43', 77, 'per person', 3
);

/* INSERT QUERY NO: 286 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_286', 'food stations', 'cat_45', 175, 'per person', 3
);

/* INSERT QUERY NO: 287 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_287', 'food stations', 'cat_46', 120, 'per person', 3
);

/* INSERT QUERY NO: 288 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_288', 'food stations', 'cat_50', 225, 'per person', 4
);

/* INSERT QUERY NO: 289 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_289', 'hors doeuvres', 'cat_08', 40, 'per person', 2
);

/* INSERT QUERY NO: 290 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_290', 'hors doeuvres', 'cat_10', 46, 'per person', 2
);

/* INSERT QUERY NO: 291 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_291', 'hors doeuvres', 'cat_16', 20, 'per person', 2
);

/* INSERT QUERY NO: 292 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_292', 'hors doeuvres', 'cat_20', 25, 'per person', 3
);

/* INSERT QUERY NO: 293 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_293', 'hors doeuvres', 'cat_21', 80, 'per person', 3
);

/* INSERT QUERY NO: 294 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_294', 'hors doeuvres', 'cat_22', 45, 'per person', 3
);

/* INSERT QUERY NO: 295 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_295', 'hors doeuvres', 'cat_23', 90, 'per person', 3
);

/* INSERT QUERY NO: 296 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_296', 'hors doeuvres', 'cat_24', 45, 'per person', 3
);

/* INSERT QUERY NO: 297 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_297', 'hors doeuvres', 'cat_25', 100, 'per person', 3
);

/* INSERT QUERY NO: 298 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_298', 'hors doeuvres', 'cat_27', 100, 'per person', 3
);

/* INSERT QUERY NO: 299 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_299', 'hors doeuvres', 'cat_28', 8, 'per person', 3
);

/* INSERT QUERY NO: 300 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_300', 'hors doeuvres', 'cat_29', 25, 'per person', 3
);

/* INSERT QUERY NO: 301 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_301', 'hors doeuvres', 'cat_35', 17, 'per person', 3
);

/* INSERT QUERY NO: 302 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_302', 'hors doeuvres', 'cat_36', 20, 'per person', 3
);

/* INSERT QUERY NO: 303 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_303', 'hors doeuvres', 'cat_40', 30, 'per person', 3
);

/* INSERT QUERY NO: 304 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_304', 'hors doeuvres', 'cat_41', 20, 'per person', 3
);

/* INSERT QUERY NO: 305 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_305', 'hors doeuvres', 'cat_43', 40, 'per person', 3
);

/* INSERT QUERY NO: 306 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_306', 'hors doeuvres', 'cat_45', 120, 'per person', 3
);

/* INSERT QUERY NO: 307 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_307', 'hors doeuvres', 'cat_46', 95, 'per person', 3
);

/* INSERT QUERY NO: 308 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_308', 'hors doeuvres', 'cat_48', 45, 'per person', 4
);

/* INSERT QUERY NO: 309 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_309', 'hors doeuvres', 'cat_50', 225, 'per person', 4
);

/* INSERT QUERY NO: 310 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_310', 'bar service', 'cat_08', 16, 'per person', 2
);

/* INSERT QUERY NO: 311 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_311', 'bar service', 'cat_10', 5, 'per person', 2
);

/* INSERT QUERY NO: 312 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_312', 'bar service', 'cat_21', 12, 'per person', 3
);

/* INSERT QUERY NO: 313 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_313', 'bar service', 'cat_22', 27, 'per person', 3
);

/* INSERT QUERY NO: 314 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_314', 'bar service', 'cat_23', 30, 'per person', 3
);

/* INSERT QUERY NO: 315 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_315', 'bar service', 'cat_24', 40, 'per person', 3
);

/* INSERT QUERY NO: 316 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_316', 'bar service', 'cat_25', 16, 'per person', 3
);

/* INSERT QUERY NO: 317 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_317', 'bar service', 'cat_27', 30, 'per person', 3
);

/* INSERT QUERY NO: 318 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_318', 'bar service', 'cat_29', 5, 'per person', 3
);

/* INSERT QUERY NO: 319 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_319', 'bar service', 'cat_35', 32, 'per person', 3
);

/* INSERT QUERY NO: 320 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_320', 'bar service', 'cat_36', 20, 'per person', 3
);

/* INSERT QUERY NO: 321 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_321', 'bar service', 'cat_39', 28, 'per person', 3
);

/* INSERT QUERY NO: 322 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_322', 'bar service', 'cat_40', 20, 'per person', 3
);

/* INSERT QUERY NO: 323 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_323', 'bar service', 'cat_41', 22, 'per person', 3
);

/* INSERT QUERY NO: 324 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_324', 'bar service', 'cat_43', 28, 'per person', 3
);

/* INSERT QUERY NO: 325 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_325', 'bar service', 'cat_45', 20, 'per person', 3
);

/* INSERT QUERY NO: 326 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_326', 'bar service', 'cat_46', 17, 'per person', 3
);

/* INSERT QUERY NO: 327 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_327', 'bar service', 'cat_48', 50, 'per person', 4
);

/* INSERT QUERY NO: 328 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_328', 'bar service', 'cat_50', 50, 'per person', 4
);
```
### Music Department
```
/* INSERT QUERY NO: 329 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_329', 'dj', 'dj_07', 550, '6 hours', 1
);

/* INSERT QUERY NO: 330 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_330', 'dj', 'dj_06', 621, '6 hours', 1
);

/* INSERT QUERY NO: 331 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_331', 'dj', 'dj_22', 650, '6 hours', 1
);

/* INSERT QUERY NO: 332 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_332', 'dj', 'dj_04', 900, '6 hours', 1
);

/* INSERT QUERY NO: 333 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_333', 'dj', 'dj_11', 1000, '6 hours', 1
);

/* INSERT QUERY NO: 334 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_334', 'dj', 'dj_19', 1000, '6 hours', 1
);

/* INSERT QUERY NO: 335 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_335', 'dj', 'dj_33', 1000, '6 hours', 1
);

/* INSERT QUERY NO: 336 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_336', 'dj', 'dj_13', 1099, '6 hours', 1
);

/* INSERT QUERY NO: 337 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_337', 'dj', 'dj_16', 1200, '6 hours', 2
);

/* INSERT QUERY NO: 338 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_338', 'dj', 'dj_21', 1200, '6 hours', 2
);

/* INSERT QUERY NO: 339 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_339', 'dj', 'dj_28', 1201, '6 hours', 2
);

/* INSERT QUERY NO: 340 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_340', 'dj', 'dj_27', 1250, '6 hours', 2
);

/* INSERT QUERY NO: 341 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_341', 'dj', 'dj_10', 1395, '6 hours', 2
);

/* INSERT QUERY NO: 342 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_342', 'dj', 'dj_31', 1395, '6 hours', 2
);

/* INSERT QUERY NO: 343 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_343', 'dj', 'dj_14', 1500, '6 hours', 2
);

/* INSERT QUERY NO: 344 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_344', 'dj', 'dj_25', 1500, '6 hours', 2
);

/* INSERT QUERY NO: 345 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_345', 'dj', 'dj_30', 1500, '6 hours', 2
);

/* INSERT QUERY NO: 346 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_346', 'dj', 'dj_34', 1795, '6 hours', 3
);

/* INSERT QUERY NO: 347 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_347', 'dj', 'dj_20', 1850, '6 hours', 3
);

/* INSERT QUERY NO: 348 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_348', 'dj', 'dj_12', 1895, '6 hours', 3
);

/* INSERT QUERY NO: 349 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_349', 'dj', 'dj_17', 1999, '6 hours', 3
);

/* INSERT QUERY NO: 350 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_350', 'dj', 'dj_08', 2000, '6 hours', 3
);

/* INSERT QUERY NO: 351 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_351', 'dj', 'dj_15', 2100, '6 hours', 3
);

/* INSERT QUERY NO: 352 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_352', 'dj', 'dj_18', 2300, '6 hours', 3
);

/* INSERT QUERY NO: 353 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_353', 'dj', 'dj_24', 2300, '6 hours', 3
);

/* INSERT QUERY NO: 354 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_354', 'dj', 'dj_02', 2500, '6 hours', 3
);

/* INSERT QUERY NO: 355 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_355', 'dj', 'dj_29', 2650, '6 hours', 4
);

/* INSERT QUERY NO: 356 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_356', 'dj', 'dj_03', 3000, '6 hours', 4
);

/* INSERT QUERY NO: 357 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_357', 'dj', 'dj_26', 3100, '6 hours', 4
);

/* INSERT QUERY NO: 358 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_358', 'dj', 'dj_09', 3200, '6 hours', 4
);

/* INSERT QUERY NO: 359 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_359', 'dj', 'dj_23', 3495, '6 hours', 4
);

/* INSERT QUERY NO: 360 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_360', 'dj', 'dj_01', 5000, '6 hours', 4
);

/* INSERT QUERY NO: 361 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_361', 'dj', 'dj_32', 5000, '6 hours', 4
);

/* INSERT QUERY NO: 362 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_362', 'dj', 'dj_35', 5499, '6 hours', 4
);

/* INSERT QUERY NO: 363 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_363', 'dj', 'dj_05', 7000, '6 hours', 4
);
```
### Photo & Video Department
```
/* INSERT QUERY NO: 364 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_364', 'video', 'vid_01', 2291, 'per service', 2
);

/* INSERT QUERY NO: 365 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_365', 'photo and video', 'vid_02', 4000, 'per service', 3
);

/* INSERT QUERY NO: 366 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_366', 'photo and video', 'vid_03', 4000, 'per service', 3
);

/* INSERT QUERY NO: 367 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_367', 'photo and video', 'vid_04', 4582, 'per service', 4
);

/* INSERT QUERY NO: 368 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_368', 'video', 'vid_05', 3818, 'per service', 3
);

/* INSERT QUERY NO: 369 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_369', 'video', 'vid_06', 3818, 'per service', 3
);

/* INSERT QUERY NO: 370 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_370', 'video', 'vid_07', 2291, 'per service', 2
);

/* INSERT QUERY NO: 371 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_371', 'video', 'vid_08', 2291, 'per service', 2
);

/* INSERT QUERY NO: 372 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_372', 'video', 'vid_09', 0, 'per service', 0
);

/* INSERT QUERY NO: 373 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_373', 'video', 'vid_10', 2291, 'per service', 2
);

/* INSERT QUERY NO: 374 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_374', 'video', 'vid_11', 3818, 'per service', 3
);

/* INSERT QUERY NO: 375 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_375', 'video', 'vid_12', 763, 'per service', 1
);

/* INSERT QUERY NO: 376 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_376', 'video', 'vid_13', 2291, 'per service', 2
);

/* INSERT QUERY NO: 377 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_377', 'video', 'vid_14', 4582, 'per service', 4
);

/* INSERT QUERY NO: 378 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_378', 'photo and video', 'vid_15', 3818, 'per service', 3
);

/* INSERT QUERY NO: 379 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_379', 'photo and video', 'vid_16', 3818, 'per service', 3
);

/* INSERT QUERY NO: 380 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_380', 'video', 'vid_17', 2291, 'per service', 2
);

/* INSERT QUERY NO: 381 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_381', 'photo and video', 'vid_18', 2291, 'per service', 2
);

/* INSERT QUERY NO: 382 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_382', 'photo and video', 'vid_19', 2291, 'per service', 2
);

/* INSERT QUERY NO: 383 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_383', 'photo and video', 'vid_20', 2291, 'per service', 2
);

/* INSERT QUERY NO: 384 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_384', 'video', 'vid_21', 2291, 'per service', 2
);

/* INSERT QUERY NO: 385 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_385', 'video', 'vid_22', 2291, 'per service', 2
);

/* INSERT QUERY NO: 386 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_386', 'video', 'vid_23', 2291, 'per service', 2
);

/* INSERT QUERY NO: 387 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_387', 'video', 'vid_24', 3818, 'per service', 3
);

/* INSERT QUERY NO: 388 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_388', 'photo and video', 'vid_25', 1500, 'per service', 1
);

/* INSERT QUERY NO: 389 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_389', 'photo', 'vid_26', 4000, 'per service', 3
);

/* INSERT QUERY NO: 390 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_390', 'photo', 'vid_27', 4000, 'per service', 3
);

/* INSERT QUERY NO: 391 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_391', 'photo', 'vid_28', 4000, 'per service', 3
);

/* INSERT QUERY NO: 392 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_392', 'photo', 'vid_29', 4000, 'per service', 3
);

/* INSERT QUERY NO: 393 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_393', 'photo', 'vid_30', 4000, 'per service', 3
);

/* INSERT QUERY NO: 394 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_394', 'photo', 'vid_31', 1500, 'per service', 1
);

/* INSERT QUERY NO: 395 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_395', 'photo', 'vid_32', 4000, 'per service', 3
);

/* INSERT QUERY NO: 396 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_396', 'photo', 'vid_33', 4000, 'per service', 3
);

/* INSERT QUERY NO: 397 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_397', 'photo and video', 'vid_34', 4000, 'per service', 3
);

/* INSERT QUERY NO: 398 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_398', 'photo and video', 'vid_35', 2500, 'per service', 2
);

/* INSERT QUERY NO: 399 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_399', 'photo and video', 'vid_36', 2500, 'per service', 2
);

/* INSERT QUERY NO: 400 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_400', 'photo', 'vid_37', 2500, 'per service', 2
);

/* INSERT QUERY NO: 401 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_401', 'photo and video', 'vid_38', 4000, 'per service', 3
);

/* INSERT QUERY NO: 402 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_402', 'photo', 'vid_39', 5000, 'per service', 4
);

/* INSERT QUERY NO: 403 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_403', 'photo and video', 'vid_40', 4000, 'per service', 3
);

/* INSERT QUERY NO: 404 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_404', 'photo', 'vid_41', 2500, 'per service', 2
);

/* INSERT QUERY NO: 405 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_405', 'photo', 'vid_42', 5000, 'per service', 4
);

/* INSERT QUERY NO: 406 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_406', 'photo', 'vid_43', 2500, 'per service', 2
);

/* INSERT QUERY NO: 407 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_407', 'photo', 'vid_44', 1500, 'per service', 1
);

/* INSERT QUERY NO: 408 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_408', 'photo and video', 'vid_45', 2500, 'per service', 2
);

/* INSERT QUERY NO: 409 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_409', 'photo', 'vid_46', 500, 'per service', 1
);

/* INSERT QUERY NO: 410 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_410', 'photo', 'vid_47', 500, 'per service', 1
);

/* INSERT QUERY NO: 411 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_411', 'photo', 'vid_48', 1500, 'per service', 1
);

/* INSERT QUERY NO: 412 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_412', 'photo', 'vid_49', 4000, 'per service', 3
);

/* INSERT QUERY NO: 413 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_413', 'photo', 'vid_50', 1500, 'per service', 1
);
```
### Hair & Makeup Department
```
/* INSERT QUERY NO: 414 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_414', 'updo', 'hmu_01', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 415 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_415', 'updo', 'hmu_01', 160, '1 per bride trial', 3
);

/* INSERT QUERY NO: 416 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_416', 'updo', 'hmu_01', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 417 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_417', 'half up', 'hmu_01', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 418 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_418', 'half up', 'hmu_01', 160, '1 per bride trial', 3
);

/* INSERT QUERY NO: 419 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_419', 'half up', 'hmu_01', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 420 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_420', 'simple', 'hmu_01', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 421 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_421', 'simple', 'hmu_01', 101, '1 per groom side', 1
);

/* INSERT QUERY NO: 422 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_422', 'simple', 'hmu_01', 115, '1 per kids', 1
);

/* INSERT QUERY NO: 423 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_423', 'traditional', 'hmu_01', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 424 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_424', 'traditional', 'hmu_01', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 425 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_425', 'traditional', 'hmu_01', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 426 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_426', 'airbrush', 'hmu_01', 240, '1 per bride', 1
);

/* INSERT QUERY NO: 427 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_427', 'airbrush', 'hmu_01', 180, '1 per bride trial', 1
);

/* INSERT QUERY NO: 428 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_428', 'traditional', 'hmu_01', 86, '1 per groom side', 1
);

/* INSERT QUERY NO: 429 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_429', 'traditional', 'hmu_01', 115, '1 per kids', 1
);

/* INSERT QUERY NO: 430 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_430', 'traditional', 'hmu_02', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 431 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_431', 'traditional', 'hmu_02', 250, '1 per bride trial', 3
);

/* INSERT QUERY NO: 432 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_432', 'traditional', 'hmu_02', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 433 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_433', 'airbrush', 'hmu_02', 380, '1 per bride', 4
);

/* INSERT QUERY NO: 434 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_434', 'airbrush', 'hmu_02', 280, '1 per bride trial', 4
);

/* INSERT QUERY NO: 435 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_435', 'traditional', 'hmu_02', 125, '1 per groom side', 1
);

/* INSERT QUERY NO: 436 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_436', 'traditional', 'hmu_02', 60, '1 per kids', 1
);

/* INSERT QUERY NO: 437 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_437', 'updo', 'hmu_03', 155, '1 per bride', 3
);

/* INSERT QUERY NO: 438 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_438', 'updo', 'hmu_03', 155, '1 per bride trial', 3
);

/* INSERT QUERY NO: 439 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_439', 'updo', 'hmu_03', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 440 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_440', 'half up', 'hmu_03', 145, '1 per bride', 2
);

/* INSERT QUERY NO: 441 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_441', 'half up', 'hmu_03', 145, '1 per bride trial', 2
);

/* INSERT QUERY NO: 442 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_442', 'half up', 'hmu_03', 115, '1 per bride side', 1
);

/* INSERT QUERY NO: 443 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_443', 'simple', 'hmu_03', 90, '1 per bride side', 1
);

/* INSERT QUERY NO: 444 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_444', 'simple', 'hmu_03', 57, '1 per groom side', 1
);

/* INSERT QUERY NO: 445 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_445', 'simple', 'hmu_03', 45, '1 per kids', 1
);

/* INSERT QUERY NO: 446 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_446', 'traditional', 'hmu_03', 110, '1 per bride', 1
);

/* INSERT QUERY NO: 447 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_447', 'traditional', 'hmu_03', 110, '1 per bride trial', 1
);

/* INSERT QUERY NO: 448 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_448', 'traditional', 'hmu_03', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 449 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_449', 'airbrush', 'hmu_03', 150, '1 per bride', 2
);

/* INSERT QUERY NO: 450 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_450', 'airbrush', 'hmu_03', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 451 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_451', 'traditional', 'hmu_03', 54, '1 per groom side', 1
);

/* INSERT QUERY NO: 452 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_452', 'traditional', 'hmu_03', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 453 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_453', 'updo', 'hmu_04', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 454 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_454', 'updo', 'hmu_04', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 455 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_455', 'updo', 'hmu_04', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 456 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_456', 'half up', 'hmu_04', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 457 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_457', 'half up', 'hmu_04', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 458 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_458', 'half up', 'hmu_04', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 459 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_459', 'simple', 'hmu_04', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 460 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_460', 'simple', 'hmu_04', 75, '1 per groom side', 1
);

/* INSERT QUERY NO: 461 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_461', 'simple', 'hmu_04', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 462 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_462', 'traditional', 'hmu_04', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 463 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_463', 'traditional', 'hmu_04', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 464 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_464', 'traditional', 'hmu_04', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 465 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_465', 'airbrush', 'hmu_04', 350, '1 per bride', 1
);

/* INSERT QUERY NO: 466 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_466', 'airbrush', 'hmu_04', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 467 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_467', 'traditional', 'hmu_04', 75, '1 per groom side', 1
);

/* INSERT QUERY NO: 468 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_468', 'traditional', 'hmu_04', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 469 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_469', 'updo', 'hmu_05', 250, '1 per bride', 3
);

/* INSERT QUERY NO: 470 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_470', 'updo', 'hmu_05', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 471 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_471', 'updo', 'hmu_05', 130, '1 per bride side', 2
);

/* INSERT QUERY NO: 472 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_472', 'half up', 'hmu_05', 250, '1 per bride', 3
);

/* INSERT QUERY NO: 473 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_473', 'half up', 'hmu_05', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 474 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_474', 'half up', 'hmu_05', 130, '1 per bride side', 2
);

/* INSERT QUERY NO: 475 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_475', 'simple', 'hmu_05', 250, '1 per bride side', 1
);

/* INSERT QUERY NO: 476 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_476', 'simple', 'hmu_05', 158, '1 per groom side', 1
);

/* INSERT QUERY NO: 477 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_477', 'simple', 'hmu_05', 125, '1 per kids', 1
);

/* INSERT QUERY NO: 478 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_478', 'traditional', 'hmu_05', 250, '1 per bride', 3
);

/* INSERT QUERY NO: 479 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_479', 'traditional', 'hmu_05', 160, '1 per bride trial', 3
);

/* INSERT QUERY NO: 480 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_480', 'traditional', 'hmu_05', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 481 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_481', 'airbrush', 'hmu_05', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 482 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_482', 'airbrush', 'hmu_05', 192, '1 per bride trial', 4
);

/* INSERT QUERY NO: 483 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_483', 'traditional', 'hmu_05', 81, '1 per groom side', 4
);

/* INSERT QUERY NO: 484 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_484', 'traditional', 'hmu_05', 75, '1 per kids', 4
);

/* INSERT QUERY NO: 485 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_485', 'updo', 'hmu_06', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 486 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_486', 'updo', 'hmu_06', 275, '1 per bride trial', 4
);

/* INSERT QUERY NO: 487 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_487', 'updo', 'hmu_06', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 488 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_488', 'half up', 'hmu_06', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 489 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_489', 'half up', 'hmu_06', 275, '1 per bride trial', 4
);

/* INSERT QUERY NO: 490 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_490', 'half up', 'hmu_06', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 491 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_491', 'simple', 'hmu_06', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 492 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_492', 'simple', 'hmu_06', 95, '1 per groom side', 1
);

/* INSERT QUERY NO: 493 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_493', 'simple', 'hmu_06', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 494 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_494', 'traditional', 'hmu_06', 229, '1 per bride', 1
);

/* INSERT QUERY NO: 495 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_495', 'traditional', 'hmu_06', 229, '1 per bride trial', 1
);

/* INSERT QUERY NO: 496 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_496', 'traditional', 'hmu_06', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 497 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_497', 'airbrush', 'hmu_06', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 498 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_498', 'airbrush', 'hmu_06', 275, '1 per bride trial', 4
);

/* INSERT QUERY NO: 499 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_499', 'traditional', 'hmu_06', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 500 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_500', 'traditional', 'hmu_06', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 501 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_501', 'updo', 'hmu_07', 265, '1 per bride', 4
);

/* INSERT QUERY NO: 502 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_502', 'updo', 'hmu_07', 175, '1 per bride trial', 3
);

/* INSERT QUERY NO: 503 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_503', 'updo', 'hmu_07', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 504 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_504', 'half up', 'hmu_07', 265, '1 per bride', 4
);

/* INSERT QUERY NO: 505 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_505', 'half up', 'hmu_07', 175, '1 per bride trial', 3
);

/* INSERT QUERY NO: 506 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_506', 'half up', 'hmu_07', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 507 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_507', 'simple', 'hmu_07', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 508 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_508', 'simple', 'hmu_07', 150, '1 per groom side', 2
);

/* INSERT QUERY NO: 509 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_509', 'simple', 'hmu_07', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 510 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_510', 'updo', 'hmu_08', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 511 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_511', 'updo', 'hmu_08', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 512 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_512', 'updo', 'hmu_08', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 513 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_513', 'half up', 'hmu_08', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 514 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_514', 'half up', 'hmu_08', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 515 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_515', 'half up', 'hmu_08', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 516 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_516', 'simple', 'hmu_08', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 517 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_517', 'simple', 'hmu_08', 76, '1 per groom side', 1
);

/* INSERT QUERY NO: 518 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_518', 'simple', 'hmu_08', 60, '1 per kids', 1
);

/* INSERT QUERY NO: 519 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_519', 'updo', 'hmu_09', 220, '1 per bride', 3
);

/* INSERT QUERY NO: 520 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_520', 'updo', 'hmu_09', 95, '1 per bride trial', 1
);

/* INSERT QUERY NO: 521 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_521', 'updo', 'hmu_09', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 522 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_522', 'half up', 'hmu_09', 220, '1 per bride', 3
);

/* INSERT QUERY NO: 523 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_523', 'half up', 'hmu_09', 95, '1 per bride trial', 1
);

/* INSERT QUERY NO: 524 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_524', 'half up', 'hmu_09', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 525 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_525', 'simple', 'hmu_09', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 526 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_526', 'simple', 'hmu_09', 76, '1 per groom side', 1
);

/* INSERT QUERY NO: 527 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_527', 'simple', 'hmu_09', 70, '1 per kids', 1
);

/* INSERT QUERY NO: 528 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_528', 'updo', 'hmu_10', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 529 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_529', 'updo', 'hmu_10', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 530 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_530', 'updo', 'hmu_10', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 531 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_531', 'half up', 'hmu_10', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 532 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_532', 'half up', 'hmu_10', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 533 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_533', 'half up', 'hmu_10', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 534 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_534', 'simple', 'hmu_10', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 535 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_535', 'simple', 'hmu_10', 95, '1 per groom side', 1
);

/* INSERT QUERY NO: 536 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_536', 'simple', 'hmu_10', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 537 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_537', 'traditional', 'hmu_10', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 538 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_538', 'traditional', 'hmu_10', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 539 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_539', 'traditional', 'hmu_10', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 540 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_540', 'airbrush', 'hmu_10', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 541 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_541', 'airbrush', 'hmu_10', 250, '1 per bride trial', 3
);

/* INSERT QUERY NO: 542 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_542', 'traditional', 'hmu_10', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 543 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_543', 'traditional', 'hmu_10', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 544 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_544', 'updo', 'hmu_11', 140, '1 per bride', 2
);

/* INSERT QUERY NO: 545 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_545', 'updo', 'hmu_11', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 546 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_546', 'updo', 'hmu_11', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 547 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_547', 'half up', 'hmu_11', 140, '1 per bride', 2
);

/* INSERT QUERY NO: 548 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_548', 'half up', 'hmu_11', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 549 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_549', 'half up', 'hmu_11', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 550 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_550', 'simple', 'hmu_11', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 551 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_551', 'simple', 'hmu_11', 63, '1 per groom side', 1
);

/* INSERT QUERY NO: 552 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_552', 'simple', 'hmu_11', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 553 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_553', 'traditional', 'hmu_11', 140, '1 per bride', 2
);

/* INSERT QUERY NO: 554 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_554', 'traditional', 'hmu_11', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 555 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_555', 'traditional', 'hmu_11', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 556 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_556', 'airbrush', 'hmu_11', 140, '1 per bride', 2
);

/* INSERT QUERY NO: 557 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_557', 'airbrush', 'hmu_11', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 558 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_558', 'traditional', 'hmu_11', 54, '1 per groom side', 1
);

/* INSERT QUERY NO: 559 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_559', 'traditional', 'hmu_11', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 560 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_560', 'traditional', 'hmu_12', 208, '1 per bride', 1
);

/* INSERT QUERY NO: 561 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_561', 'traditional', 'hmu_12', 167, '1 per bride trial', 1
);

/* INSERT QUERY NO: 562 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_562', 'traditional', 'hmu_12', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 563 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_563', 'airbrush', 'hmu_12', 250, '1 per bride', 3
);

/* INSERT QUERY NO: 564 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_564', 'airbrush', 'hmu_12', 200, '1 per bride trial', 3
);

/* INSERT QUERY NO: 565 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_565', 'traditional', 'hmu_12', 54, '1 per groom side', 1
);

/* INSERT QUERY NO: 566 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_566', 'traditional', 'hmu_12', 65, '1 per kids', 1
);

/* INSERT QUERY NO: 567 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_567', 'updo', 'hmu_13', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 568 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_568', 'updo', 'hmu_13', 400, '1 per bride trial', 4
);

/* INSERT QUERY NO: 569 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_569', 'updo', 'hmu_13', 300, '1 per bride side', 4
);

/* INSERT QUERY NO: 570 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_570', 'half up', 'hmu_13', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 571 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_571', 'half up', 'hmu_13', 400, '1 per bride trial', 4
);

/* INSERT QUERY NO: 572 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_572', 'half up', 'hmu_13', 300, '1 per bride side', 4
);

/* INSERT QUERY NO: 573 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_573', 'simple', 'hmu_13', 300, '1 per bride side', 4
);

/* INSERT QUERY NO: 574 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_574', 'simple', 'hmu_13', 189, '1 per groom side', 1
);

/* INSERT QUERY NO: 575 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_575', 'simple', 'hmu_13', 150, '1 per kids', 1
);

/* INSERT QUERY NO: 576 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_576', 'updo', 'hmu_14', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 577 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_577', 'updo', 'hmu_14', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 578 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_578', 'updo', 'hmu_14', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 579 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_579', 'half up', 'hmu_14', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 580 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_580', 'half up', 'hmu_14', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 581 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_581', 'half up', 'hmu_14', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 582 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_582', 'simple', 'hmu_14', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 583 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_583', 'simple', 'hmu_14', 50, '1 per groom side', 1
);

/* INSERT QUERY NO: 584 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_584', 'simple', 'hmu_14', 40, '1 per kids', 1
);

/* INSERT QUERY NO: 585 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_585', 'traditional', 'hmu_14', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 586 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_586', 'traditional', 'hmu_14', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 587 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_587', 'traditional', 'hmu_14', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 588 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_588', 'airbrush', 'hmu_14', 360, '1 per bride', 1
);

/* INSERT QUERY NO: 589 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_589', 'airbrush', 'hmu_14', 180, '1 per bride trial', 1
);

/* INSERT QUERY NO: 590 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_590', 'traditional', 'hmu_14', 50, '1 per groom side', 1
);

/* INSERT QUERY NO: 591 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_591', 'traditional', 'hmu_14', 40, '1 per kids', 1
);

/* INSERT QUERY NO: 592 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_592', 'updo', 'hmu_15', 375, '1 per bride', 4
);

/* INSERT QUERY NO: 593 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_593', 'updo', 'hmu_15', 375, '1 per bride trial', 1
);

/* INSERT QUERY NO: 594 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_594', 'updo', 'hmu_15', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 595 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_595', 'half up', 'hmu_15', 375, '1 per bride', 4
);

/* INSERT QUERY NO: 596 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_596', 'half up', 'hmu_15', 375, '1 per bride trial', 1
);

/* INSERT QUERY NO: 597 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_597', 'half up', 'hmu_15', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 598 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_598', 'simple', 'hmu_15', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 599 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_599', 'simple', 'hmu_15', 79, '1 per groom side', 1
);

/* INSERT QUERY NO: 600 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_600', 'simple', 'hmu_15', 63, '1 per kids', 1
);

/* INSERT QUERY NO: 601 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_601', 'updo', 'hmu_16', 175, '1 per bride', 3
);

/* INSERT QUERY NO: 602 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_602', 'updo', 'hmu_16', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 603 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_603', 'updo', 'hmu_16', 80, '1 per bride side', 1
);

/* INSERT QUERY NO: 604 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_604', 'half up', 'hmu_16', 175, '1 per bride', 3
);

/* INSERT QUERY NO: 605 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_605', 'half up', 'hmu_16', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 606 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_606', 'half up', 'hmu_16', 80, '1 per bride side', 1
);

/* INSERT QUERY NO: 607 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_607', 'simple', 'hmu_16', 75, '1 per bride side', 1
);

/* INSERT QUERY NO: 608 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_608', 'simple', 'hmu_16', 47, '1 per groom side', 1
);

/* INSERT QUERY NO: 609 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_609', 'simple', 'hmu_16', 38, '1 per kids', 1
);

/* INSERT QUERY NO: 610 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_610', 'traditional', 'hmu_16', 175, '1 per bride', 3
);

/* INSERT QUERY NO: 611 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_611', 'traditional', 'hmu_16', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 612 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_612', 'traditional', 'hmu_16', 85, '1 per bride side', 1
);

/* INSERT QUERY NO: 613 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_613', 'airbrush', 'hmu_16', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 614 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_614', 'airbrush', 'hmu_16', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 615 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_615', 'traditional', 'hmu_16', 46, '1 per groom side', 1
);

/* INSERT QUERY NO: 616 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_616', 'traditional', 'hmu_16', 43, '1 per kids', 1
);

/* INSERT QUERY NO: 617 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_617', 'traditional', 'hmu_17', 450, '1 per bride', 4
);

/* INSERT QUERY NO: 618 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_618', 'traditional', 'hmu_17', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 619 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_619', 'traditional', 'hmu_17', 95, '1 per bride side', 1
);

/* INSERT QUERY NO: 620 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_620', 'airbrush', 'hmu_17', 450, '1 per bride', 4
);

/* INSERT QUERY NO: 621 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_621', 'airbrush', 'hmu_17', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 622 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_622', 'traditional', 'hmu_17', 51, '1 per groom side', 1
);

/* INSERT QUERY NO: 623 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_623', 'traditional', 'hmu_17', 48, '1 per kids', 1
);

/* INSERT QUERY NO: 624 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_624', 'traditional', 'hmu_18', 270, '1 per bride', 1
);

/* INSERT QUERY NO: 625 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_625', 'traditional', 'hmu_18', 280, '1 per bride trial', 1
);

/* INSERT QUERY NO: 626 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_626', 'traditional', 'hmu_18', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 627 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_627', 'airbrush', 'hmu_18', 225, '1 per bride', 3
);

/* INSERT QUERY NO: 628 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_628', 'airbrush', 'hmu_18', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 629 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_629', 'traditional', 'hmu_18', 25, '1 per groom side', 1
);

/* INSERT QUERY NO: 630 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_630', 'traditional', 'hmu_18', 25, '1 per kids', 1
);

/* INSERT QUERY NO: 631 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_631', 'traditional', 'hmu_19', 375, '1 per bride', 4
);

/* INSERT QUERY NO: 632 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_632', 'traditional', 'hmu_19', 175, '1 per bride trial', 3
);

/* INSERT QUERY NO: 633 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_633', 'traditional', 'hmu_19', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 634 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_634', 'airbrush', 'hmu_19', 375, '1 per bride', 4
);

/* INSERT QUERY NO: 635 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_635', 'airbrush', 'hmu_19', 175, '1 per bride trial', 3
);

/* INSERT QUERY NO: 636 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_636', 'traditional', 'hmu_19', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 637 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_637', 'traditional', 'hmu_19', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 638 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_638', 'updo', 'hmu_20', 299, '1 per bride', 4
);

/* INSERT QUERY NO: 639 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_639', 'updo', 'hmu_20', 299, '1 per bride trial', 4
);

/* INSERT QUERY NO: 640 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_640', 'updo', 'hmu_20', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 641 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_641', 'half up', 'hmu_20', 299, '1 per bride', 4
);

/* INSERT QUERY NO: 642 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_642', 'half up', 'hmu_20', 299, '1 per bride trial', 4
);

/* INSERT QUERY NO: 643 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_643', 'half up', 'hmu_20', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 644 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_644', 'simple', 'hmu_20', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 645 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_645', 'simple', 'hmu_20', 63, '1 per groom side', 1
);

/* INSERT QUERY NO: 646 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_646', 'simple', 'hmu_20', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 647 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_647', 'traditional', 'hmu_20', 299, '1 per bride', 4
);

/* INSERT QUERY NO: 648 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_648', 'traditional', 'hmu_20', 199, '1 per bride trial', 3
);

/* INSERT QUERY NO: 649 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_649', 'traditional', 'hmu_20', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 650 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_650', 'airbrush', 'hmu_20', 299, '1 per bride', 4
);

/* INSERT QUERY NO: 651 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_651', 'airbrush', 'hmu_20', 199, '1 per bride trial', 3
);

/* INSERT QUERY NO: 652 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_652', 'traditional', 'hmu_20', 65, '1 per groom side', 1
);

/* INSERT QUERY NO: 653 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_653', 'traditional', 'hmu_20', 60, '1 per kids', 1
);

/* INSERT QUERY NO: 654 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_654', 'traditional', 'hmu_21', 250, '1 per traditional', 3
);

/* INSERT QUERY NO: 655 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_655', 'traditional', 'hmu_21', 200, '1 per traditional', 3
);

/* INSERT QUERY NO: 656 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_656', 'traditional', 'hmu_21', 150, '1 per traditional', 2
);

/* INSERT QUERY NO: 657 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_657', 'airbrush', 'hmu_21', 300, '1 per airbrush', 1
);

/* INSERT QUERY NO: 658 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_658', 'airbrush', 'hmu_21', 240, '1 per airbrush', 1
);

/* INSERT QUERY NO: 659 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_659', 'traditional', 'hmu_21', 81, '1 per traditional', 1
);

/* INSERT QUERY NO: 660 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_660', 'traditional', 'hmu_21', 75, '1 per traditional', 1
);

/* INSERT QUERY NO: 661 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_661', 'updo', 'hmu_22', 260, '1 per bride', 4
);

/* INSERT QUERY NO: 662 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_662', 'updo', 'hmu_22', 130, '1 per bride trial', 2
);

/* INSERT QUERY NO: 663 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_663', 'updo', 'hmu_22', 130, '1 per bride side', 2
);

/* INSERT QUERY NO: 664 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_664', 'half up', 'hmu_22', 260, '1 per bride', 4
);

/* INSERT QUERY NO: 665 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_665', 'half up', 'hmu_22', 130, '1 per bride trial', 2
);

/* INSERT QUERY NO: 666 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_666', 'half up', 'hmu_22', 130, '1 per bride side', 2
);

/* INSERT QUERY NO: 667 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_667', 'simple', 'hmu_22', 130, '1 per bride side', 2
);

/* INSERT QUERY NO: 668 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_668', 'simple', 'hmu_22', 45, '1 per groom side', 1
);

/* INSERT QUERY NO: 669 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_669', 'simple', 'hmu_22', 45, '1 per kids', 1
);

/* INSERT QUERY NO: 670 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_670', 'updo', 'hmu_23', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 671 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_671', 'updo', 'hmu_23', 200, '1 per bride trial', 1
);

/* INSERT QUERY NO: 672 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_672', 'updo', 'hmu_23', 200, '1 per bride side', 1
);

/* INSERT QUERY NO: 673 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_673', 'half up', 'hmu_23', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 674 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_674', 'half up', 'hmu_23', 200, '1 per bride trial', 1
);

/* INSERT QUERY NO: 675 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_675', 'half up', 'hmu_23', 200, '1 per bride side', 1
);

/* INSERT QUERY NO: 676 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_676', 'simple', 'hmu_23', 200, '1 per bride side', 3
);

/* INSERT QUERY NO: 677 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_677', 'simple', 'hmu_23', 126, '1 per groom side', 1
);

/* INSERT QUERY NO: 678 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_678', 'simple', 'hmu_23', 100, '1 per kids', 1
);

/* INSERT QUERY NO: 679 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_679', 'traditional', 'hmu_23', 125, '1 per bride', 1
);

/* INSERT QUERY NO: 680 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_680', 'traditional', 'hmu_23', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 681 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_681', 'traditional', 'hmu_23', 125, '1 per bride side', 1
);

/* INSERT QUERY NO: 682 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_682', 'airbrush', 'hmu_23', 150, '1 per bride', 1
);

/* INSERT QUERY NO: 683 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_683', 'airbrush', 'hmu_23', 150, '1 per bride trial', 1
);

/* INSERT QUERY NO: 684 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_684', 'traditional', 'hmu_23', 68, '1 per groom side', 1
);

/* INSERT QUERY NO: 685 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_685', 'traditional', 'hmu_23', 63, '1 per kids', 1
);

/* INSERT QUERY NO: 686 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_686', 'traditional', 'hmu_24', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 687 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_687', 'traditional', 'hmu_24', 200, '1 per bride trial', 3
);

/* INSERT QUERY NO: 688 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_688', 'traditional', 'hmu_24', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 689 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_689', 'airbrush', 'hmu_24', 420, '1 per bride', 1
);

/* INSERT QUERY NO: 690 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_690', 'airbrush', 'hmu_24', 240, '1 per bride trial', 1
);

/* INSERT QUERY NO: 691 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_691', 'traditional', 'hmu_24', 200, '1 per groom side', 3
);

/* INSERT QUERY NO: 692 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_692', 'traditional', 'hmu_24', 170, '1 per kids', 3
);

/* INSERT QUERY NO: 693 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_693', 'updo', 'hmu_25', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 694 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_694', 'updo', 'hmu_25', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 695 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_695', 'updo', 'hmu_25', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 696 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_696', 'half up', 'hmu_25', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 697 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_697', 'half up', 'hmu_25', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 698 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_698', 'half up', 'hmu_25', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 699 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_699', 'simple', 'hmu_25', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 700 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_700', 'simple', 'hmu_25', 76, '1 per groom side', 1
);

/* INSERT QUERY NO: 701 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_701', 'simple', 'hmu_25', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 702 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_702', 'updo', 'hmu_26', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 703 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_703', 'updo', 'hmu_26', 275, '1 per bride trial', 4
);

/* INSERT QUERY NO: 704 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_704', 'updo', 'hmu_26', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 705 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_705', 'half up', 'hmu_26', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 706 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_706', 'half up', 'hmu_26', 275, '1 per bride trial', 4
);

/* INSERT QUERY NO: 707 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_707', 'half up', 'hmu_26', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 708 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_708', 'simple', 'hmu_26', 175, '1 per bride side', 1
);

/* INSERT QUERY NO: 709 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_709', 'simple', 'hmu_26', 200, '1 per groom side', 3
);

/* INSERT QUERY NO: 710 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_710', 'simple', 'hmu_26', 100, '1 per kids', 1
);

/* INSERT QUERY NO: 711 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_711', 'updo', 'hmu_27', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 712 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_712', 'updo', 'hmu_27', 300, '1 per bride trial', 4
);

/* INSERT QUERY NO: 713 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_713', 'updo', 'hmu_27', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 714 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_714', 'half up', 'hmu_27', 300, '1 per bride', 4
);

/* INSERT QUERY NO: 715 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_715', 'half up', 'hmu_27', 300, '1 per bride trial', 4
);

/* INSERT QUERY NO: 716 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_716', 'half up', 'hmu_27', 160, '1 per bride side', 3
);

/* INSERT QUERY NO: 717 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_717', 'simple', 'hmu_27', 160, '1 per bride side', 1
);

/* INSERT QUERY NO: 718 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_718', 'simple', 'hmu_27', 120, '1 per groom side', 1
);

/* INSERT QUERY NO: 719 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_719', 'simple', 'hmu_27', 80, '1 per kids', 1
);

/* INSERT QUERY NO: 720 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_720', 'traditional', 'hmu_28', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 721 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_721', 'traditional', 'hmu_28', 170, '1 per bride trial', 3
);

/* INSERT QUERY NO: 722 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_722', 'traditional', 'hmu_28', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 723 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_723', 'airbrush', 'hmu_28', 240, '1 per bride', 1
);

/* INSERT QUERY NO: 724 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_724', 'airbrush', 'hmu_28', 204, '1 per bride trial', 1
);

/* INSERT QUERY NO: 725 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_725', 'traditional', 'hmu_28', 65, '1 per groom side', 1
);

/* INSERT QUERY NO: 726 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_726', 'traditional', 'hmu_28', 60, '1 per kids', 1
);

/* INSERT QUERY NO: 727 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_727', 'traditional', 'hmu_29', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 728 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_728', 'traditional', 'hmu_29', 200, '1 per bride trial', 3
);

/* INSERT QUERY NO: 729 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_729', 'traditional', 'hmu_29', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 730 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_730', 'airbrush', 'hmu_29', 480, '1 per bride', 1
);

/* INSERT QUERY NO: 731 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_731', 'airbrush', 'hmu_29', 240, '1 per bride trial', 1
);

/* INSERT QUERY NO: 732 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_732', 'traditional', 'hmu_29', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 733 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_733', 'traditional', 'hmu_29', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 734 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_734', 'updo', 'hmu_30', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 735 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_735', 'updo', 'hmu_30', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 736 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_736', 'updo', 'hmu_30', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 737 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_737', 'half up', 'hmu_30', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 738 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_738', 'half up', 'hmu_30', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 739 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_739', 'half up', 'hmu_30', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 740 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_740', 'simple', 'hmu_30', 95, '1 per bride side', 1
);

/* INSERT QUERY NO: 741 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_741', 'simple', 'hmu_30', 60, '1 per groom side', 1
);

/* INSERT QUERY NO: 742 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_742', 'simple', 'hmu_30', 90, '1 per kids', 1
);

/* INSERT QUERY NO: 743 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_743', 'traditional', 'hmu_30', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 744 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_744', 'traditional', 'hmu_30', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 745 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_745', 'traditional', 'hmu_30', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 746 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_746', 'airbrush', 'hmu_30', 275, '1 per bride', 4
);

/* INSERT QUERY NO: 747 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_747', 'airbrush', 'hmu_30', 125, '1 per bride trial', 1
);

/* INSERT QUERY NO: 748 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_748', 'traditional', 'hmu_30', 150, '1 per groom side', 2
);

/* INSERT QUERY NO: 749 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_749', 'traditional', 'hmu_30', 90, '1 per kids', 1
);

/* INSERT QUERY NO: 750 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_750', 'traditional', 'hmu_31', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 751 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_751', 'traditional', 'hmu_31', 200, '1 per bride trial', 3
);

/* INSERT QUERY NO: 752 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_752', 'traditional', 'hmu_31', 120, '1 per bride side', 1
);

/* INSERT QUERY NO: 753 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_753', 'airbrush', 'hmu_31', 250, '1 per bride', 3
);

/* INSERT QUERY NO: 754 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_754', 'airbrush', 'hmu_31', 250, '1 per bride trial', 3
);

/* INSERT QUERY NO: 755 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_755', 'traditional', 'hmu_31', 65, '1 per groom side', 1
);

/* INSERT QUERY NO: 756 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_756', 'traditional', 'hmu_31', 65, '1 per kids', 1
);

/* INSERT QUERY NO: 757 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_757', 'traditional', 'hmu_32', 254, '1 per bride', 4
);

/* INSERT QUERY NO: 758 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_758', 'traditional', 'hmu_32', 254, '1 per bride trial', 1
);

/* INSERT QUERY NO: 759 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_759', 'traditional', 'hmu_32', 254, '1 per bride side', 4
);

/* INSERT QUERY NO: 760 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_760', 'airbrush', 'hmu_32', 305, '1 per bride', 1
);

/* INSERT QUERY NO: 761 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_761', 'airbrush', 'hmu_32', 305, '1 per bride trial', 1
);

/* INSERT QUERY NO: 762 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_762', 'traditional', 'hmu_32', 137, '1 per groom side', 1
);

/* INSERT QUERY NO: 763 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_763', 'traditional', 'hmu_32', 127, '1 per kids', 1
);

/* INSERT QUERY NO: 764 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_764', 'updo', 'hmu_33', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 765 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_765', 'updo', 'hmu_33', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 766 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_766', 'updo', 'hmu_33', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 767 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_767', 'half up', 'hmu_33', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 768 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_768', 'half up', 'hmu_33', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 769 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_769', 'half up', 'hmu_33', 175, '1 per bride side', 3
);

/* INSERT QUERY NO: 770 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_770', 'simple', 'hmu_33', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 771 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_771', 'simple', 'hmu_33', 95, '1 per groom side', 1
);

/* INSERT QUERY NO: 772 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_772', 'simple', 'hmu_33', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 773 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_773', 'traditional', 'hmu_33', 350, '1 per bride', 4
);

/* INSERT QUERY NO: 774 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_774', 'traditional', 'hmu_33', 200, '1 per bride trial', 3
);

/* INSERT QUERY NO: 775 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_775', 'traditional', 'hmu_33', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 776 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_776', 'airbrush', 'hmu_33', 400, '1 per bride', 4
);

/* INSERT QUERY NO: 777 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_777', 'airbrush', 'hmu_33', 250, '1 per bride trial', 3
);

/* INSERT QUERY NO: 778 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_778', 'traditional', 'hmu_33', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 779 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_779', 'traditional', 'hmu_33', 75, '1 per kids', 1
);

/* INSERT QUERY NO: 780 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_780', 'updo', 'hmu_34', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 781 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_781', 'updo', 'hmu_34', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 782 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_782', 'updo', 'hmu_34', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 783 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_783', 'half up', 'hmu_34', 200, '1 per bride', 3
);

/* INSERT QUERY NO: 784 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_784', 'half up', 'hmu_34', 100, '1 per bride trial', 1
);

/* INSERT QUERY NO: 785 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_785', 'half up', 'hmu_34', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 786 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_786', 'simple', 'hmu_34', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 787 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_787', 'simple', 'hmu_34', 63, '1 per groom side', 1
);

/* INSERT QUERY NO: 788 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_788', 'simple', 'hmu_34', 50, '1 per kids', 1
);

/* INSERT QUERY NO: 789 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_789', 'updo', 'hmu_35', 225, '1 per bride', 3
);

/* INSERT QUERY NO: 790 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_790', 'updo', 'hmu_35', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 791 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_791', 'updo', 'hmu_35', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 792 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_792', 'half up', 'hmu_35', 225, '1 per bride', 3
);

/* INSERT QUERY NO: 793 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_793', 'half up', 'hmu_35', 150, '1 per bride trial', 2
);

/* INSERT QUERY NO: 794 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_794', 'half up', 'hmu_35', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 795 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_795', 'simple', 'hmu_35', 100, '1 per bride side', 1
);

/* INSERT QUERY NO: 796 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_796', 'simple', 'hmu_35', 25, '1 per groom side', 1
);

/* INSERT QUERY NO: 797 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_797', 'simple', 'hmu_35', 25, '1 per kids', 1
);

/* INSERT QUERY NO: 798 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_798', 'traditional', 'hmu_35', 230, '1 per bride', 3
);

/* INSERT QUERY NO: 799 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_799', 'traditional', 'hmu_35', 160, '1 per bride trial', 3
);

/* INSERT QUERY NO: 800 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_800', 'traditional', 'hmu_35', 150, '1 per bride side', 2
);

/* INSERT QUERY NO: 801 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_801', 'airbrush', 'hmu_35', 260, '1 per bride', 4
);

/* INSERT QUERY NO: 802 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_802', 'airbrush', 'hmu_35', 192, '1 per bride trial', 1
);

/* INSERT QUERY NO: 803 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_803', 'traditional', 'hmu_35', 81, '1 per groom side', 1
);

/* INSERT QUERY NO: 804 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_804', 'traditional', 'hmu_35', 75, '1 per kids', 1
);
```
### Dress & Attire Department
```
/* INSERT QUERY NO: 805 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_805', 'dress - belobed by casablance bridal', 'att_01', 1750, '1 per person', 4
);

/* INSERT QUERY NO: 806 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_806', 'dress - madison james by allure bridals', 'att_02', 1750, '1 per person', 4
);

/* INSERT QUERY NO: 807 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_807', 'dress - disney fairy tale weddings', 'att_02', 2250, '1 per person', 4
);

/* INSERT QUERY NO: 808 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_808', 'dress - stacees amy wedding dress', 'att_03', 616, '1 per person', 4
);

/* INSERT QUERY NO: 809 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_809', 'dress - stacees isabella wedding dress', 'att_03', 537, '1 per person', 4
);

/* INSERT QUERY NO: 810 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_810', 'dress - stacees annie wedding dress', 'att_03', 417, '1 per person', 3
);

/* INSERT QUERY NO: 811 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_811', 'dress - stacees elaine wedding dress', 'att_03', 537, '1 per person', 4
);

/* INSERT QUERY NO: 812 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_812', 'dress - luxury lace long sleeves slit wedding dresses', 'att_03', 350, '1 per person', 3
);

/* INSERT QUERY NO: 813 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_813', 'dress - luxury lace off-the-shoulder wedding dresses with chapel train', 'att_04', 320, '1 per person', 2
);

/* INSERT QUERY NO: 814 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_814', 'dress - trumpet luxury v-neck long sleeve wedding dresses with appliques lace', 'att_04', 350, '1 per person', 3
);

/* INSERT QUERY NO: 815 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_815', 'dress - ball-gown v-neck sleeveless tulle sweep train wedding dress with appliques', 'att_04', 320, '1 per person', 2
);

/* INSERT QUERY NO: 816 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_816', 'dress - mermaid one-shoulder luxury satin wedding dresses with beading', 'att_04', 199, '1 per person', 1
);

/* INSERT QUERY NO: 817 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_817', 'dress - lace mermaid classic wedding dresses with elastic satin', 'att_04', 249, '1 per person', 2
);

/* INSERT QUERY NO: 818 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_818', 'dress - sequined off-the-shoulder long sleeves wedding dresses with detachable train', 'att_05', 170, '1 per person', 1
);

/* INSERT QUERY NO: 819 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_819', 'dress - boho tulle off-the-shoulder slit wedding dresses', 'att_05', 189, '1 per person', 1
);

/* INSERT QUERY NO: 820 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_820', 'dress - ball-gown/princess v-neck court train tulle wedding dress with beading', 'att_05', 324, '1 per person', 2
);

/* INSERT QUERY NO: 821 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_821', 'dress - trumpet scoop court train lace wedding dress', 'att_06', 244, '1 per person', 2
);

/* INSERT QUERY NO: 822 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_822', 'dress - princess v-neck court train tulle wedding dress', 'att_06', 404, '1 per person', 3
);

/* INSERT QUERY NO: 823 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_823', 'dress - mermaid v-neck court train lace wedding dress', 'att_06', 329, '1 per person', 2
);

/* INSERT QUERY NO: 824 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_824', 'dress - a-line v-neck asymmetrical tulle wedding dress with cascading ruffles pleated', 'att_05', 104, '1 per person', 1
);

/* INSERT QUERY NO: 825 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_825', 'dress - a-line v-neck sweep train chiffon wedding dress with beading sequins', 'att_05', 224, '1 per person', 2
);

/* INSERT QUERY NO: 826 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_826', 'dress - allure bridesmaids', 'att_02', 225, '1 per person', 2
);

/* INSERT QUERY NO: 827 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_827', 'dress - birdy grey', 'att_06', 75, '1 per person', 1
);

/* INSERT QUERY NO: 828 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_828', 'dress - aw bridal', 'att_07', 125, '1 per person', 1
);

/* INSERT QUERY NO: 829 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_829', 'dress - morilee by madeline gardner ', 'att_08', 175, '1 per person', 1
);

/* INSERT QUERY NO: 830 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_830', 'dress - scarf neck one-shoulder stretch satin mermaid dress with slight train - cs109', 'att_09', 349, '1 per person', 3
);

/* INSERT QUERY NO: 831 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_831', 'dress - long puff sleeve v-neck trumpet gown', 'att_09', 275, '1 per person', 2
);

/* INSERT QUERY NO: 832 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_832', 'dress - christina wu celebration', 'att_10', 175, '1 per person', 1
);

/* INSERT QUERY NO: 833 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_833', 'dress - kennedy blue', 'att_11', 125, '1 per person', 1
);

/* INSERT QUERY NO: 834 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_834', 'dress - jada couture mother of the bride by jasmine', 'att_12', 550, '1 per person', 4
);

/* INSERT QUERY NO: 835 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_835', 'dress - cameron  blake', 'att_13', 550, '1 per person', 4
);

/* INSERT QUERY NO: 836 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_836', 'dress - ivonne d. ', 'att_13', 700, '1 per person', 4
);

/* INSERT QUERY NO: 837 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_837', 'dress - jada couture mother of the bride by jasmine', 'att_12', 450, '1 per person', 3
);

/* INSERT QUERY NO: 838 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_838', 'dress - jada couture mother of the bride by jasmine', 'att_12', 450, '1 per person', 3
);

/* INSERT QUERY NO: 839 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_839', 'dress - jada couture mother of the bride by jasmine', 'att_12', 450, '1 per person', 3
);

/* INSERT QUERY NO: 840 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_840', 'tuxedo - Pronto Uomo Platinum Modern Fit Suit Separates Tuxedo Coat', 'att_14', 249.99, '1 per person', 1
);

/* INSERT QUERY NO: 841 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_841', 'suit - JOE Joseph Abboud Slim Fit Suit Separates Coat', 'att_14', 249.99, '1 per person', 1
);

/* INSERT QUERY NO: 842 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_842', 'suit - dark navy tonal plaid stretch wool suit', 'att_15', 1325, '1 per person', 4
);

/* INSERT QUERY NO: 843 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_843', 'suit - navy lightweight wool suit', 'att_16', 3495, '1 per person', 4
);

/* INSERT QUERY NO: 844 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_844', 'suit - stretch wool black suit', 'att_17', 206, '1 per person', 1
);

/* INSERT QUERY NO: 845 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_845', 'tuxedo - black watch tartan tuxedo', 'att_17', 488, '1 per person', 3
);

/* INSERT QUERY NO: 846 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_846', 'tuxedo - light blue shawl jacket tuxedo', 'att_17', 469, '1 per person', 2
);

/* INSERT QUERY NO: 847 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_847', 'tuxedo - night sky velvet tuxedo', 'att_17', 469, '1 per person', 2
);

/* INSERT QUERY NO: 848 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_848', 'tuxedo - split tartan jacket tuxedo', 'att_17', 469, '1 per person', 2
);

/* INSERT QUERY NO: 849 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_849', 'tuxedo - notch lapel tuxedo', 'att_17', 495, '1 per person', 3
);

/* INSERT QUERY NO: 850 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_850', 'tuxedo - peak lapel tuxedo', 'att_17', 550, '1 per person', 4
);

/* INSERT QUERY NO: 851 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_851', 'tuxedo - shawl collar tuxedo', 'att_17', 550, '1 per person', 4
);

/* INSERT QUERY NO: 852 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_852', 'suit - grey suit', 'att_17', 375, '1 per person', 1
);

/* INSERT QUERY NO: 853 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_853', 'suit - navy suit', 'att_17', 375, '1 per person', 1
);

/* INSERT QUERY NO: 854 */
INSERT INTO Products(product_id, product_name, vendor_id, price_unit, unit_vol, price_ce)
VALUES
(
'prod_854', 'suit - charcoal suit', 'att_17', 475, '1 per person', 3
);
```
### VI. Vendor Sustainability Table
### Flower Department
```
DROP TABLE IF EXISTS Flowers_Vendor_Sustainability;
CREATE TABLE Flowers_Vendor_Sustainability (
    id_fvs INT PRIMARY KEY NOT NULL,
    vendor_id VARCHAR(255) NOT NULL,
    local_sf_supply_chain INT,
    artificial_flowers_in_portfolio INT,
    FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id)
);

DROP TABLE IF EXISTS Flower_Season_Style;
CREATE TABLE Flower_Season_Style (
    id_fss INT PRIMARY KEY NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    flower_season VARCHAR(255),
    flower_style VARCHAR(255),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

#inserting for team 1 until 8663
INSERT INTO Flower_Season_Style (id_fss, product_id, flower_season, flower_style)
VALUES
(1, "prod_184", "spring", "romantic"),
(2, "prod_185", "spring", "glamorous"),
(3, "prod_186", "fall", "eclectic"),
(4, "prod_187", "summer", "tropical"),
(5, "prod_188", "winter", "country"),
(6, "prod_189", "fall", "vintage"),
(7, "prod_190", "spring", "modern"),
(8, "prod_191", "fall", "classic"),
(9, "prod_192", "spring", "warm"),
(10, "prod_193", "winter", "eclectic"),
(11, "prod_194", "fall", "country"),
(12, "prod_195", "summer", "tropical"),
(13, "prod_196", "spring", "romantic"),
(14, "prod_197", "winter", "romantic"),
(15, "prod_198", "spring", "glamorous"),
(16, "prod_199", "fall", "classic"),
(17, "prod_200", "spring", "modern"),
(18, "prod_201", "spring", "eclectic"),
(19, "prod_202", "summer", "tropical"),
(20, "prod_203", "summer", "romantic"),
(21, "prod_204", "spring", "modern"),
(22, "prod_205", "spring", "glamorous"),
(23, "prod_206", "spring", "eclectic"),
(24, "prod_207", "spring", "country"),
(25, "prod_208", "fall", "warm"),
(26, "prod_209", "summer", "classic"),
(27, "prod_210", "spring", "rustic"),
(28, "prod_211", "winter", "nautical"),
(29, "prod_212", "summer", "preppy"),
(30, "prod_213", "spring", "classic"),
(31, "prod_214", "spring", "tropical"),
(32, "prod_215", "summer", "country"),
(33, "prod_216", "spring", "elegant"),
(34, "prod_217", "winter", "preppy"),
(35, "prod_218", "fall", "vintage"),
(36, "prod_219", "summer", "romantic"),
(37, "prod_220", "spring", "glamorous"),
(38, "prod_221", "winter", "romantic"),
(39, "prod_222", "winter", "modern"),
(40, "prod_223", "spring", "vintage"),
(41, "prod_224", "fall", "rustic"),
(42, "prod_225", "fall", "classic");

INSERT INTO Flowers_Vendor_Sustainability (id_fvs, vendor_id, local_sf_supply_chain, artificial_flowers_in_portfolio)
VALUES
(1, "flo_01", 1, 0),
(2, "flo_02", 0, 1),
(3, "flo_03", 1, 0),
(4, "flo_04", 1, 0),
(5, "flo_05", 1, 0),
(6, "flo_06", 1, 0),
(7, "flo_07", 0, 0),
(8, "flo_08", 1, 0),
(9, "flo_09", 1, 0),
(10, "flo_10", 0, 1),
(11, "flo_11", 1, 0),
(12, "flo_12", 1, 0),
(13, "flo_13", 1, 0),
(14, "flo_14", 0, 0),
(15, "flo_15", 1, 0),
(16, "flo_16", 1, 0),
(17, "flo_17", 1, 0),
(18, "flo_18", 1, 0),
(19, "flo_19", 0, 1),
(20, "flo_20", 1, 0),
(21, "flo_21", 1, 0),
(22, "flo_22", 0, 0),
(23, "flo_23", 1, 0),
(24, "flo_24", 0, 1),
(25, "flo_25", 1, 0),
(26, "flo_26", 1, 0),
(27, "flo_27", 1, 0),
(28, "flo_28", 0, 1),
(29, "flo_29", 1, 0),
(30, "flo_30", 0, 1),
(31, "flo_31", 1, 0),
(32, "flo_32", 1, 1),
(33, "flo_33", 1, 0),
(34, "flo_34", 1, 1),
(35, "flo_35", 0, 1),
(36, "flo_36", 1, 0),
(37, "flo_37", 0, 1),
(38, "flo_38", 1, 1),
(39, "flo_39", 1, 0),
(40, "flo_40", 1, 0),
(41, "flo_41", 1, 0),
(42, "flo_42", 1, 0);
```
### Music Department
```
#Creating Popularity Table
DROP TABLE IF EXISTS Popularity;
CREATE TABLE Popularity (
    vendor_id VARCHAR(20),
    Popularity_ID VARCHAR(20),
    RATINGS DECIMAL(2,1),
    AWARDS INT,
    PRIMARY KEY (Popularity_ID),
    FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id)
);  
  

DROP TABLE IF EXISTS Sustainability;
CREATE TABLE Sustainability (
    Sustainability_ID INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id VARCHAR(20),
    equip_ec INT,
    avg_usage_hours INT,
    total_ec DECIMAL(4,2),
    number_equipment INT,
    sustainability INT,
    FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id)
);

#inserting dj data until row 8767
INSERT INTO Popularity (vendor_id, Popularity_ID, RATINGS, AWARDS)
VALUES 
('dj_01', 1, 4.9, 5),
('dj_02', 2, 4.9, 8),
('dj_03', 3, 5, 2),
('dj_04', 4, 5, 1),
('dj_05', 5, 5, 5),
('dj_06', 6, 5, 0),
('dj_07', 7, 4.5, 2),
('dj_08', 8, 4.5, 7),
('dj_09', 9, 5, 6),
('dj_10', 10, 5, 7),
('dj_11', 11, 5, 7),
('dj_12', 12, 5, 2),
('dj_13', 13, 5, 1),
('dj_14', 14, 5, 2),
('dj_15', 15, 4.8, 1),
('dj_16', 16, 4.8, 7),
('dj_17', 17, 4.3, 7),
('dj_18', 18, 4.9, 2),
('dj_19', 19, 5, 3),
('dj_20', 20, 5, 0),
('dj_21', 21, 3, 0),
('dj_22', 22, 5, 0),
('dj_23', 23, 5, 0),
('dj_24', 24, 4.9, 1),
('dj_25', 25, 4.8, 1),
('dj_26', 26, 5, 0),
('dj_27', 27, 4.7, 2),
('dj_28', 28, 5, 1),
('dj_29', 29, 4.9, 5),
('dj_30', 30, 4.9, 3),
('dj_31', 31, 5, 7),
('dj_32', 32, 5, 3),
('dj_33', 33, 5, 3),
('dj_34', 34, 5, 7),
('dj_35', 35, 5, 3);

INSERT INTO Sustainability (vendor_id, equip_ec, avg_usage_hours, total_ec, number_equipment, sustainability)
VALUES 
('dj_01', 2491, 6, 14.95, 8, 0),
('dj_02', 2079, 6, 12.47, 10, 1),
('dj_03', 2489, 6, 14.93, 8, 0),
('dj_04', 2077, 6, 14.26, 4, 0),
('dj_05', 2377, 6, 14.26, 8, 0),
('dj_06', 2065, 6, 12.39, 6, 1),
('dj_07', 2177, 6, 13.06, 7, 1),
('dj_08', 2027, 6, 12.16, 4, 1),
('dj_09', 2179, 6, 13.07, 5, 1),
('dj_10', 3077, 6, 18.46, 3, 0),
('dj_11', 2479, 6, 14.87, 4, 0),
('dj_12', 2377, 6, 14.26, 3, 0),
('dj_13', 2067, 6, 12.4, 4, 1),
('dj_14', 2179, 6, 13.07, 4, 1),
('dj_15', 2067, 6, 12.4, 3, 1),
('dj_16', 2477, 6, 14.86, 6, 0),
('dj_17', 2177, 6, 14.35, 4, 1),
('dj_18', 2391, 6, 12.39, 3, 0),
('dj_19', 2065, 6, 14.27, 5, 1),
('dj_20', 2379, 6, 13.07, 4, 0),
('dj_21', 2179, 6, 12.39, 3, 1),
('dj_22', 2065, 6, 13.07, 4, 1),
('dj_23', 2179, 6, 12.39, 3, 1),
('dj_24', 2065, 6, 12.39, 3, 1),
('dj_25', 2065, 6, 12.39, 3, 1),
('dj_26', 2065, 6, 13.07, 4, 1),
('dj_27', 2179, 6, 14.87, 6, 1),
('dj_28', 2479, 6, 14.27, 5, 0),
('dj_29', 2379, 6, 14.27, 5, 0),
('dj_30', 2379, 6, 12.39, 3, 0),
('dj_31', 2065, 6, 13.07, 4, 1),
('dj_32', 2179, 6, 14.27, 5, 1),
('dj_33', 2379, 6, 14.27, 6, 0),
('dj_34', 2379, 6, 12.39, 3, 0),
('dj_35', 2065, 6, 14.27, 6, 1);
```
### Hair & Makeup Department
```
#Team 7 tables and inserts until 8897

-- Optional table about sustainability for hair and makeup department
/* CREATE TABLE */
DROP TABLE IF EXISTS hmusustainability;
CREATE TABLE hmusustainability(
sustainability_id smallint unsigned NOT NULL,
vendor_id VARCHAR(255) NOT NULL,
product VARCHAR(255),
environmental VARCHAR(255),
ethical VARCHAR(255),
economic VARCHAR(255),
PRIMARY KEY (sustainability_id),
FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id)
);

/* INSERT QUERY NO: 1 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
1, 'hmu_01', 'luminous silk foundation', 'recyclable packaging', '"paraben-free"', 'fair trade and locally sourced'
);

/* INSERT QUERY NO: 2 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
2, 'hmu_01', 'scattered light glitter eyeshadow', 'refillable packaging', '"cruelty-free"', ''
);

/* INSERT QUERY NO: 3 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
3, 'hmu_01', 'ultra precise brow pencil', 'eco friendly packaging', '"cruelty-free and vegan-free"', 'fair trade'
);

/* INSERT QUERY NO: 4 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
4, 'hmu_07', 'gentle makeup remover', 'recyclable packaging', '"cruelty-free"', ''
);

/* INSERT QUERY NO: 5 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
5, 'hmu_07', 'lady lush liquid lipstick', '', '"cruelty-free"', 'locally sourced'
);

/* INSERT QUERY NO: 6 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
6, 'hmu_11', 'major arcana matte lipstick', '', '"cruelty-free"', 'locally sourced'
);

/* INSERT QUERY NO: 7 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
7, 'hmu_12', 'cosmetics and skincare', '', '"paraben-free"', ''
);

/* INSERT QUERY NO: 8 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
8, 'hmu_24', 'W3LL People bio stick foundation', 'recyclable packaging', '"cruelty-free"', 'locally sourced'
);

/* INSERT QUERY NO: 9 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
9, 'hmu_25', 'cosmetic pouch', 'Reusable EVA BPA free', '', ''
);

/* INSERT QUERY NO: 10 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
10, 'hmu_26', 'eyelashes', 'recyclable packaging', '"paraben-free"', ''
);

/* INSERT QUERY NO: 11 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
11, 'hmu_26', 'color related', 'recyclable packaging', '"cruelty-free"', 'fair trade and locally sourced'
);

/* INSERT QUERY NO: 12 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
12, 'hmu_27', 'beauty products', '', '', ''
);

/* INSERT QUERY NO: 13 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
13, 'hmu_30', 'beauty products', '', '"cruelty-free and vegan-free"', ''
);

/* INSERT QUERY NO: 14 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
14, 'hmu_31', 'airbrush tanning', '', '"cruelty-free and vegan-free"', ''
);

/* INSERT QUERY NO: 15 */
INSERT INTO hmusustainability(sustainability_id, vendor_id, product, environmental, ethical, economic)
VALUES
(
15, 'hmu_34', 'beauty products', 'recyclable packaging', '"cruelty-free and paraben-free"', ''
);
```
### Dress & Attire Department
```
#Team 8 data until 9287
/* Showing results for dress.xlsx */
DROP TABLE IF EXISTS dress;

/* CREATE TABLE */
CREATE TABLE dress(
dress_id INT NOT NULL auto_increment,
silhouette VARCHAR(100),
neckline VARCHAR(100),
sleeve VARCHAR(100),
fabric VARCHAR(100),
product_id VARCHAR(100),
PRIMARY KEY (dress_id),
FOREIGN KEY (product_id) REFERENCES products(product_id)
);


/* Showing results for attire.xlsx */
DROP TABLE IF EXISTS attire;

/* CREATE TABLE */
CREATE TABLE attire(
attire_id int not null auto_increment,
color VARCHAR(100),
tie VARCHAR(100),
number_of_buttons DOUBLE,
lapel_style VARCHAR(100),
product_id VARCHAR(100),
PRIMARY KEY (attire_id),
FOREIGN KEY (product_id) REFERENCES Products(product_id)
);



/* INSERT QUERY NO: 1 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
1, 'a-line', 'v-neck', 'without sleeve', 'tulle', 'prod_805'
);

/* INSERT QUERY NO: 2 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
2, 'a-line', 'sweetheart', 'without sleeve', 'specialty', 'prod_806'
);

/* INSERT QUERY NO: 3 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
3, 'a-line', 'sweetheart', 'illusion', 'tulle', 'prod_807'
);

/* INSERT QUERY NO: 4 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
4, 'ball gown', 'v-neck', 'without sleeve', 'organza', 'prod_808'
);

/* INSERT QUERY NO: 5 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
5, 'princess', 'illusion', 'long', 'tulle', 'prod_809'
);

/* INSERT QUERY NO: 6 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
6, 'a-line', 'v-neck', 'without sleeve', 'lace', 'prod_810'
);

/* INSERT QUERY NO: 7 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
7, 'trumpet', 'v-neck', 'long', 'tulle', 'prod_811'
);

/* INSERT QUERY NO: 8 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
8, 'a-line', 'illusion', 'long', 'lace', 'prod_812'
);

/* INSERT QUERY NO: 9 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
9, 'mermaid', 'off-the-shoulder', 'without sleeve', 'lace', 'prod_813'
);

/* INSERT QUERY NO: 10 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
10, 'trumpet', 'v-neck', 'long', 'lace', 'prod_814'
);

/* INSERT QUERY NO: 11 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
11, 'ball gown', 'v-neck', 'without sleeve', 'tulle', 'prod_815'
);

/* INSERT QUERY NO: 12 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
12, 'mermaid', 'one-shoulder', 'without sleeve', 'satin', 'prod_816'
);

/* INSERT QUERY NO: 13 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
13, 'mermaid', 'v-neck', 'without sleeve', 'satin', 'prod_817'
);

/* INSERT QUERY NO: 14 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
14, 'trumpet', 'off-the-shoulder', 'long', 'sequined', 'prod_818'
);

/* INSERT QUERY NO: 15 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
15, 'ball gown', 'off-the-shoulder', 'without sleeve', 'tulle', 'prod_819'
);

/* INSERT QUERY NO: 16 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
16, 'ball gown', 'v-neck', 'without sleeve', 'tulle', 'prod_820'
);

/* INSERT QUERY NO: 17 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
17, 'trumpet', 'scoop', 'without sleeve', 'lace', 'prod_821'
);

/* INSERT QUERY NO: 18 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
18, 'princess', 'v-neck', 'long', 'tulle', 'prod_822'
);

/* INSERT QUERY NO: 19 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
19, 'mermaid', 'v-neck', 'long', 'lace', 'prod_823'
);

/* INSERT QUERY NO: 20 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
20, 'a-line', 'v-neck', 'without sleeve', 'tulle', 'prod_824'
);

/* INSERT QUERY NO: 21 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
21, 'a-line', 'v-neck', 'without sleeve', 'chiffon', 'prod_825'
);

/* INSERT QUERY NO: 22 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
22, 'a-line', 'sweetheart', 'without sleeve', 'lace', 'prod_826'
);

/* INSERT QUERY NO: 23 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
23, 'a-line', 'v-neck', 'without sleeve', 'chiffon', 'prod_827'
);

/* INSERT QUERY NO: 24 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
24, 'a-line', 'v-neck', 'without sleeve', 'satin', 'prod_828'
);

/* INSERT QUERY NO: 25 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
25, 'a-line', 'v-neck', 'without sleeve', 'chiffon', 'prod_829'
);

/* INSERT QUERY NO: 26 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
26, 'mermaid', 'one-shoulder', 'without sleeve', 'satin', 'prod_830'
);

/* INSERT QUERY NO: 27 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
27, 'mermaid', 'v-neck', 'long', 'chiffon', 'prod_831'
);

/* INSERT QUERY NO: 28 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
28, 'a-line', 'off-the-shoulder', 'without sleeve', 'chiffon', 'prod_832'
);

/* INSERT QUERY NO: 29 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
29, 'a-line', 'one-shoulder', 'short', 'chiffon', 'prod_833'
);

/* INSERT QUERY NO: 30 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
30, 'mermaid', 'off-the-shoulder', '3/4 sleeve', 'lace', 'prod_834'
);

/* INSERT QUERY NO: 31 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
31, 'a-line', 'illusion', 'illusion', 'chiffon', 'prod_835'
);

/* INSERT QUERY NO: 32 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
32, 'mermaid', 'illusion', 'without sleeve', 'tulle', 'prod_836'
);

/* INSERT QUERY NO: 33 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
33, 'a-line', 'illusion', '3/4 sleeve', 'lace', 'prod_837'
);

/* INSERT QUERY NO: 34 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
34, 'mermaid', 'v-neck', 'without sleeve', 'tulle', 'prod_838'
);

/* INSERT QUERY NO: 35 */
INSERT INTO dress(dress_id, silhouette, neckline, sleeve, fabric, product_id)
VALUES
(
35, 'mermaid', 'bateau', 'cap', 'chiffon', 'prod_839'
);

--
--
--
--
/* attire table starts here*/ 
-- 
--
--
--


/* INSERT QUERY NO: 1 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
1, 'dark blue', 'bow tie', 1, 'shawl', 'prod_840'
);

/* INSERT QUERY NO: 2 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
2, 'grey', 'bow tie', 2, 'notch', 'prod_841'
);

/* INSERT QUERY NO: 3 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
3, 'dark navy', 'necktie', 2, 'notch', 'prod_842'
);

/* INSERT QUERY NO: 4 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
4, 'navy', 'necktie', 2, 'notch', 'prod_843'
);

/* INSERT QUERY NO: 5 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
5, 'black', 'necktie', 2, 'notch', 'prod_844'
);

/* INSERT QUERY NO: 6 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
6, 'green and blue', 'bow tie', 1, 'satin peak', 'prod_845'
);

/* INSERT QUERY NO: 7 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
7, 'light blue', 'bow tie', 1, 'shawl', 'prod_846'
);

/* INSERT QUERY NO: 8 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
8, 'night sky', 'bow tie', 1, 'satin shawl', 'prod_847'
);

/* INSERT QUERY NO: 9 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
9, 'multicolor', 'bow tie', 1, 'satin peak', 'prod_848'
);

/* INSERT QUERY NO: 10 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
10, 'black', 'bow tie', 1, 'notch', 'prod_849'
);

/* INSERT QUERY NO: 11 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
11, 'black', 'bow tie', 1, 'satin peak', 'prod_850'
);

/* INSERT QUERY NO: 12 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
12, 'black', 'bow tie', 1, 'shawl', 'prod_851'
);

/* INSERT QUERY NO: 13 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
13, 'grey', 'necktie', 2, 'notch', 'prod_852'
);

/* INSERT QUERY NO: 14 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
14, 'navy', 'necktie', 2, 'notch', 'prod_853'
);

/* INSERT QUERY NO: 15 */
INSERT INTO attire(attire_id, color, tie, number_of_buttons, lapel_style, product_id)
VALUES
(
15, 'charcoal', 'necktie', 2, 'notch', 'prod_854'
);
```
### Invitation Department
```
#Team 11 data until 9709

/* Showing results for Invitations (2).xlsx */

/* CREATE TABLE */
DROP TABLE IF EXISTS inv_mailing;
CREATE TABLE inv_mailing(
mailing_id INT UNSIGNED NOT NULL,
name VARCHAR(100),
cost_05_lb_nat DOUBLE,
cost_05_lb_int DOUBLE,
sustainable DOUBLE,
mailing_url VARCHAR(100),
PRIMARY KEY (mailing_id)
);

/* INSERT QUERY NO: 1 */
INSERT INTO inv_mailing(mailing_id, name, cost_05_lb_nat, cost_05_lb_int, sustainable, mailing_url)
VALUES
(
1, 'Flat Export', 9.48, 11.88, 1, '"""https://www.easyship.com/couriers/flat-export"""'
);

/* INSERT QUERY NO: 2 */
INSERT INTO inv_mailing(mailing_id, name, cost_05_lb_nat, cost_05_lb_int, sustainable, mailing_url)
VALUES
(
2, 'DHL Express', 7.43, 20, 0, '"""https://www.dhl.com/us-en/home.html?locale=true"""'
);

/* INSERT QUERY NO: 3 */
INSERT INTO inv_mailing(mailing_id, name, cost_05_lb_nat, cost_05_lb_int, sustainable, mailing_url)
VALUES
(
3, 'USPS', 1.5, 3, 1, '"""https://www.usps.com/"""'
);

/* INSERT QUERY NO: 4 */
INSERT INTO inv_mailing(mailing_id, name, cost_05_lb_nat, cost_05_lb_int, sustainable, mailing_url)
VALUES
(
4, 'Fedex', 26.36, 67.28, 0, '"""https://www.fedex.com/en-us/home.html"""'
);






/* Showing results for Untitled spreadsheet.xlsx */
/* Showing results for Untitled spreadsheet (1).xlsx */

/* CREATE TABLE */
DROP TABLE IF EXISTS inv_characteristics;
CREATE TABLE inv_characteristics(
characteristic_id INT UNSIGNED NOT NULL,
product_category VARCHAR(100),
product_type VARCHAR(100),
price_i DOUBLE,
weight_i_lb DOUBLE,
sustainability_rating DOUBLE,
mailing_id INT UNSIGNED,
vendor_id VARCHAR(100),
PRIMARY KEY (characteristic_id),
FOREIGN KEY (vendor_id) REFERENCES Vendors(vendor_id),
FOREIGN KEY (mailing_id) REFERENCES inv_mailing(mailing_id)
);

/* INSERT QUERY NO: 1 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
1, 'Envelope Liner', 'Recycled ', 289.00, 0.007, 1, 1, 'inv_01'
);

/* INSERT QUERY NO: 2 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
3, 'Envelope Liner', 'Recycled ', 289.00, 0.007, 1, 3, 'inv_02'
);

/* INSERT QUERY NO: 3 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
4, 'Envelope Liner', 'Recycled ', 289.00, 0.007, 1, 4, 'inv_03'
);

/* INSERT QUERY NO: 4 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
6, 'Envelope Liner', 'Matte text weight', 259.00, 0.005, 0, 2, 'inv_04'
);

/* INSERT QUERY NO: 5 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
7, 'Envelope Liner', 'Matte text weight', 259.00, 0.005, 0, 3, 'inv_05'
);

/* INSERT QUERY NO: 6 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
9, 'RSVP', 'Recycled', 150.00, 0.013, 1, 1, 'inv_06'
);

/* INSERT QUERY NO: 7 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
11, 'RSVP', 'Recycled', 150.00, 0.013, 1, 3, 'inv_07'
);

/* INSERT QUERY NO: 8 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
12, 'RSVP', 'Recycled', 150.00, 0.013, 1, 4, 'inv_08'
);

/* INSERT QUERY NO: 9 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
14, 'RSVP', 'Recycled', 700.00, 0.013, 1, 2, 'inv_09'
);

/* INSERT QUERY NO: 10 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
16, 'RSVP', 'Recycled', 700.00, 0.013, 1, 4, 'inv_10'
);

/* INSERT QUERY NO: 11 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
18, 'RSVP', 'Signature', 285.00, 0.012, 0, 2, 'inv_11'
);

/* INSERT QUERY NO: 12 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
20, 'RSVP', 'Signature', 285.00, 0.012, 0, 4, 'inv_12'
);

/* INSERT QUERY NO: 13 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
22, 'RSVP', 'Pearlescent', 169.00, 0.011, 0, 2, 'inv_13'
);

/* INSERT QUERY NO: 14 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
23, 'RSVP', 'Pearlescent', 169.00, 0.011, 0, 3, 'inv_14'
);

/* INSERT QUERY NO: 15 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
25, 'Invitation', 'Signature', 189.00, 0.016, 0, NULL, 'inv_15'
);

/* INSERT QUERY NO: 16 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
28, 'Invitation', 'Signature', 289.00, 0.016, 0, NULL, 'inv_16'
);

/* INSERT QUERY NO: 17 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
29, 'Invitation', 'Recycled', 259.00, 0.018, 1, NULL, 'inv_17'
);

/* INSERT QUERY NO: 18 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
31, 'Invitation', 'Recycled', 229.00, 0.018, 1, 2, 'inv_18'
);

/* INSERT QUERY NO: 19 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
33, 'Invitation', 'Pearlescent', 229.00, 0.014, 0, 4, 'inv_19'
);

/* INSERT QUERY NO: 20 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
37, 'Invitation', 'Recycled', 289.00, 0.018, 1, 1, 'inv_20'
);

/* INSERT QUERY NO: 21 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
39, 'Invitation', 'Recycled', 289.00, 0.018, 1, 3, 'inv_21'
);

/* INSERT QUERY NO: 22 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
41, 'Invitation', 'Premium', 289.00, 0.020, 1, 1, 'inv_22'
);

/* INSERT QUERY NO: 23 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
43, 'Invitation', 'Premium', 289.00, 0.020, 1, 3, 'inv_23'
);

/* INSERT QUERY NO: 24 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
46, 'Invitation', 'Double Thick', 199.00, 0.020, 1, 2, 'inv_24'
);

/* INSERT QUERY NO: 25 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
49, 'Invitation', 'Triple Thick', 69.00, 0.020, 1, 1, 'inv_25'
);

/* INSERT QUERY NO: 26 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
50, 'Invitation', 'Triple Thick', 69.00, 0.020, 1, 2, 'inv_26'
);

/* INSERT QUERY NO: 27 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
52, 'Invitation', 'Triple Thick', 69.00, 0.020, 1, 4, 'inv_27'
);

/* INSERT QUERY NO: 28 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
55, 'Envelope Liner', 'Recycled', 249.00, 0.008, 1, NULL, 'inv_28'
);

/* INSERT QUERY NO: 29 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
58, 'Envelope Liner', 'Recycled', 188.00, 0.008, 1, 1, 'inv_29'
);

/* INSERT QUERY NO: 30 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
60, 'Invitation', 'Standard', 188.00, 0.015, 0, 3, 'inv_30'
);

/* INSERT QUERY NO: 31 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
63, 'RSVP', 'Standard', 207.20, 0.014, 0, 2, 'inv_31'
);

/* INSERT QUERY NO: 32 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
64, 'RSVP', 'Standard', 207.20, 0.014, 0, 3, 'inv_32'
);

/* INSERT QUERY NO: 33 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
67, 'Envelope Liner', 'Standard', 226.00, 0.004, 0, 2, 'inv_33'
);

/* INSERT QUERY NO: 34 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
68, 'Envelope Liner', 'Standard', 226.00, 0.004, 0, 3, 'inv_34'
);

/* INSERT QUERY NO: 35 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
71, 'Place Card', 'Signature', 2468.00, 0.008, 0, NULL, 'inv_35'
);

/* INSERT QUERY NO: 36 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
72, 'Place Card', 'Recycled', 268.00, 0.009, 1, NULL, 'inv_36'
);

/* INSERT QUERY NO: 37 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
73, 'Place Card', 'Pearlescent', 157.00, 0.007, 0, 1, 'inv_37'
);

/* INSERT QUERY NO: 38 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
75, 'Table Number', 'Recycled', 157.00, 0.012, 1, 3, 'inv_38'
);

/* INSERT QUERY NO: 39 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
77, 'Table Number', 'Recycled', 199.00, 0.012, 1, NULL, 'inv_39'
);

/* INSERT QUERY NO: 40 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
78, 'RSVP', 'Digital', 151.00, 0, 1, NULL, 'inv_40'
);

/* INSERT QUERY NO: 41 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
80, 'Wedding Program', 'Recycled', 199.00, 0.013, 1, NULL, 'inv_41'
);

/* INSERT QUERY NO: 42 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
81, 'Wedding Program', 'Signature', 69.00, 0.012, 0, 1, 'inv_42'
);

/* INSERT QUERY NO: 43 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
84, 'Menu', 'Recycled', 69.00, 0.019, 1, 4, 'inv_43'
);

/* INSERT QUERY NO: 44 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
85, 'Menu', 'Recycled', 952.00, 0.019, 1, NULL, 'inv_44'
);

/* INSERT QUERY NO: 45 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
87, 'Menu', 'Signature', 269.00, 0.017, 0, 2, 'inv_45'
);

/* INSERT QUERY NO: 46 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
89, 'Invitation', 'Digital', 269.00, 0, 1, 4, 'inv_46'
);

/* INSERT QUERY NO: 47 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
92, 'Table Number', 'Triple Thick', 2508.00, 0.02, 1, NULL, 'inv_47'
);

/* INSERT QUERY NO: 48 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
94, 'Wedding Program', 'Premium', 50.00, 0.019, 1, NULL, 'inv_48'
);

/* INSERT QUERY NO: 49 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
95, 'Wedding Program', 'Double Thick', 219.00, 0.019, 1, NULL, 'inv_49'
);

/* INSERT QUERY NO: 50 */
INSERT INTO inv_characteristics(characteristic_id, product_category, product_type, price_i, weight_i_lb, sustainability_rating, mailing_id, vendor_id)
VALUES
(
98, 'Menu', 'Triple Thick', 799.00, 0.019, 1, NULL, 'inv_50'
);
```
### Venue Department
```
#Data for Team 2 venues until 10819
DROP TABLE IF EXISTS ven_settings;
CREATE TABLE ven_settings(
    venset_id VARCHAR(10) NOT NULL,
    ven_type VARCHAR(50),
    ven_set1 VARCHAR(50),
    ven_set2 VARCHAR(50),
    ven_set3 VARCHAR(50),
    ven_set4 VARCHAR(50),
    ven_set5 VARCHAR(50),
    product_id VARCHAR(10),
    PRIMARY KEY (venset_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

DROP TABLE IF EXISTS ven_amenities;
CREATE TABLE ven_amenities(
    ven_amenity VARCHAR(10) NOT NULL,
    ven_ceremony_area BOOL,
    ven_dress_room BOOL,
    ven_handicap_access BOOL,
    ven_indoor BOOL,
    ven_outdoor BOOL,
    ven_liability_ins BOOL,
    ven_covered_outd BOOL,
    ven_accommodation BOOL,    
    product_id VARCHAR(10),
    PRIMARY KEY (ven_amenity),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

DROP TABLE IF EXISTS ven_ceremonies;
CREATE TABLE ven_ceremonies(
    ven_ceremony VARCHAR(10) NOT NULL,
    ven_civil_u BOOL,
    ven_nonrelig_cerem BOOL,
    ven_relig_cerem BOOL,
    product_id VARCHAR(10),
    PRIMARY KEY (ven_ceremony),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

/* INSERT QUERY NO: 1 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset01', 'ballroom', 'country club', '', '', '', '', 'prod_001'
);

/* INSERT QUERY NO: 2 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset02', 'backyard', 'ballroom', 'barn', 'beach', 'garden', 'vineyard & winery', 'prod_002'
);

/* INSERT QUERY NO: 3 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset03', 'ballroom', 'beach', 'waterfront', 'tented', 'vineyard & winery', '', 'prod_003'
);

/* INSERT QUERY NO: 4 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset04', 'ballroom', 'barn', 'beach', 'waterfront', 'garden', 'hotel', 'prod_004'
);

/* INSERT QUERY NO: 5 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset05', 'barn', 'farm & ranch', 'historic venue', 'mountain', '', '', 'prod_005'
);

/* INSERT QUERY NO: 6 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset06', 'country club', 'restaurant', 'trees', '', '', '', 'prod_006'
);

/* INSERT QUERY NO: 7 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset07', 'backyard', 'ballroom', 'garden', 'rooftop', '', '', 'prod_007'
);

/* INSERT QUERY NO: 8 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset08', 'ballroom', 'beach', 'waterfront', 'historic venue', 'restaurant ', '', 'prod_008'
);

/* INSERT QUERY NO: 9 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset09', 'ballroom', 'country club', 'historic venue', '', '', '', 'prod_009'
);

/* INSERT QUERY NO: 10 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset10', 'ballroom', 'industrial & warehouse', 'loft', 'restaurant ', '', '', 'prod_010'
);

/* INSERT QUERY NO: 11 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset11', 'ballroom', 'country club', '', '', '', '', 'prod_011'
);

/* INSERT QUERY NO: 12 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset12', 'ballroom', 'castle', 'garden', 'historic venue', 'hotel', '', 'prod_012'
);

/* INSERT QUERY NO: 13 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset13', 'ballroom', 'historic venue', 'hotel', 'restaurant', 'rooftop', '', 'prod_013'
);

/* INSERT QUERY NO: 14 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset14', 'backyard', 'ballroom', 'barn', 'farm & ranch', 'garden', 'historic venue', 'prod_014'
);

/* INSERT QUERY NO: 15 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset15', 'country club', 'park', 'tented', '', '', '', 'prod_015'
);

/* INSERT QUERY NO: 16 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset16', 'ballroom', 'beach', 'waterfront', 'country club', 'cruise', 'historic venue', 'prod_016'
);

/* INSERT QUERY NO: 17 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset17', 'backyard', 'ballroom', 'country club', 'garden', 'historic venue', '', 'prod_017'
);

/* INSERT QUERY NO: 18 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset18', 'backyard', 'ballroom', 'barn', 'farm & ranch', 'garden', 'hotel', 'prod_018'
);

/* INSERT QUERY NO: 19 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset19', 'backyard', 'barn', 'farm & ranch', 'beach', 'garden', 'park', 'prod_019'
);

/* INSERT QUERY NO: 20 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset20', 'ballroom', 'historic venue', 'hotel', '', '', '', 'prod_020'
);

/* INSERT QUERY NO: 21 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset21', 'ballroom', 'hotel', 'restaurant', 'rooftop', '', '', 'prod_021'
);

/* INSERT QUERY NO: 22 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset22', 'hotel', '', '', '', '', '', 'prod_022'
);

/* INSERT QUERY NO: 23 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset23', 'ballroom', 'country club', '', '', '', '', 'prod_023'
);

/* INSERT QUERY NO: 24 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset24', 'ballroom', 'garden', 'hotel', 'restaurant', '', '', 'prod_024'
);

/* INSERT QUERY NO: 25 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset25', 'ballroom', 'barn', 'historic venue', 'hotel', 'mountain', '', 'prod_025'
);

/* INSERT QUERY NO: 26 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset26', 'country club', 'ballroom', 'barn', 'farm & ranch', 'beach', 'garden', 'prod_026'
);

/* INSERT QUERY NO: 27 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset27', 'historic camp', 'barn', 'historic venue', 'mountain', 'park', 'trees', 'prod_027'
);

/* INSERT QUERY NO: 28 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset28', 'museum', 'ballroom', 'garden', 'museum', '', '', 'prod_028'
);

/* INSERT QUERY NO: 29 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset29', 'restaurant', 'garden', 'restaurant', '', '', '', 'prod_029'
);

/* INSERT QUERY NO: 30 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset30', 'university club', 'backyard', 'ballroom', 'historic venue', 'library', '', 'prod_030'
);

/* INSERT QUERY NO: 31 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset31', 'farm', 'barn', 'farm & ranch', 'garden', 'trees', '', 'prod_031'
);

/* INSERT QUERY NO: 32 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset32', 'farm', 'barn', 'farm & ranch', 'garden', 'mountains', 'trees', 'prod_032'
);

/* INSERT QUERY NO: 33 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset33', 'ballroom', 'hotel', 'restaurant', '', '', '', 'prod_033'
);

/* INSERT QUERY NO: 34 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset34', 'ballroom', 'castle', 'religious settings', 'historic venue', 'mountains', 'park', 'prod_034'
);

/* INSERT QUERY NO: 35 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset35', 'ballroom', 'hotel', '', '', '', '', 'prod_035'
);

/* INSERT QUERY NO: 36 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset36', 'ballroom', 'garden', '', '', '', '', 'prod_036'
);

/* INSERT QUERY NO: 37 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset37', 'ballroom', 'estate', 'vineyar & winery', 'barn', 'garden', 'mountain', 'prod_037'
);

/* INSERT QUERY NO: 38 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset38', 'ballroom', 'historic venue', 'restaurant', '', '', '', 'prod_038'
);

/* INSERT QUERY NO: 39 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset39', 'estate', 'garden', 'historic venue', 'mountain', 'trees', 'vineyard & winery', 'prod_039'
);

/* INSERT QUERY NO: 40 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset40', 'ballroom', 'barn', 'garden', 'hotel', '', '', 'prod_040'
);

/* INSERT QUERY NO: 41 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset41', 'beach', 'waterfront', 'brewery & distillery', 'historic venue', 'vineyard & winery', '', 'prod_041'
);

/* INSERT QUERY NO: 42 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset42', 'ballroom', 'garden', 'historic venue', 'rooftop', 'trees', '', 'prod_042'
);

/* INSERT QUERY NO: 43 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset43', 'garden ', 'park', 'trees', '', '', '', 'prod_043'
);

/* INSERT QUERY NO: 44 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset44', 'ballroom', '', '', '', '', '', 'prod_044'
);

/* INSERT QUERY NO: 45 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset45', 'garden', 'museum', 'rooftop', '', '', '', 'prod_045'
);

/* INSERT QUERY NO: 46 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset46', 'ballroom', 'historic venue', 'hotel', 'restaurant', 'trees', '', 'prod_046'
);

/* INSERT QUERY NO: 47 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset47', 'ballroom', 'historic venue', '', '', '', '', 'prod_047'
);

/* INSERT QUERY NO: 48 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset48', 'beach', 'waterfront', 'garden', 'historic venue', 'rooftop', '', 'prod_048'
);

/* INSERT QUERY NO: 49 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset49', 'beach', 'waterfront', 'country club', 'garden', 'restaurant', 'tented', 'prod_049'
);

/* INSERT QUERY NO: 50 */
INSERT INTO ven_settings(venset_id, ven_type, ven_set1, ven_set2, ven_set3, ven_set4, ven_set5, product_id)
VALUES
(
'venset50', 'ballroom', 'hotel', '', '', '', '', 'prod_050'
);

/* INSERT QUERY NO: 1 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame01', 1, 1, 1, 1, 1, 1, 0, 0, 'prod_001'
);

/* INSERT QUERY NO: 2 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame02', 1, 1, 1, 1, 1, 1, 0, 1, 'prod_002'
);

/* INSERT QUERY NO: 3 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame03', 1, 1, 1, 1, 1, 1, 1, 1, 'prod_003'
);

/* INSERT QUERY NO: 4 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame04', 1, 1, 1, 1, 1, 0, 1, 1, 'prod_004'
);

/* INSERT QUERY NO: 5 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame05', 1, 1, 1, 1, 1, 1, 0, 0, 'prod_005'
);

/* INSERT QUERY NO: 6 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame06', 1, 1, 1, 1, 1, 0, 1, 0, 'prod_006'
);

/* INSERT QUERY NO: 7 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame07', 1, 1, 1, 1, 1, 0, 0, 0, 'prod_007'
);

/* INSERT QUERY NO: 8 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame08', 0, 0, 1, 1, 0, 0, 0, 0, 'prod_008'
);

/* INSERT QUERY NO: 9 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame09', 1, 0, 1, 1, 1, 0, 0, 0, 'prod_009'
);

/* INSERT QUERY NO: 10 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame10', 1, 0, 1, 1, 1, 1, 0, 0, 'prod_010'
);

/* INSERT QUERY NO: 11 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame11', 1, 1, 0, 1, 1, 0, 0, 0, 'prod_011'
);

/* INSERT QUERY NO: 12 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame12', 1, 0, 1, 1, 0, 0, 0, 1, 'prod_012'
);

/* INSERT QUERY NO: 13 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame13', 1, 0, 1, 1, 0, 0, 0, 1, 'prod_013'
);

/* INSERT QUERY NO: 14 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame14', 1, 1, 1, 1, 1, 1, 0, 0, 'prod_014'
);

/* INSERT QUERY NO: 15 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame15', 1, 1, 1, 1, 1, 0, 1, 0, 'prod_015'
);

/* INSERT QUERY NO: 16 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame16', 0, 0, 1, 1, 1, 1, 0, 0, 'prod_016'
);

/* INSERT QUERY NO: 17 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame17', 1, 1, 1, 1, 1, 0, 1, 1, 'prod_017'
);

/* INSERT QUERY NO: 18 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame18', 1, 1, 1, 1, 1, 1, 0, 1, 'prod_018'
);

/* INSERT QUERY NO: 19 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame19', 1, 1, 1, 1, 1, 0, 0, 1, 'prod_019'
);

/* INSERT QUERY NO: 20 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame20', 0, 0, 1, 1, 0, 0, 0, 1, 'prod_020'
);

/* INSERT QUERY NO: 21 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame21', 0, 1, 1, 1, 1, 0, 0, 1, 'prod_021'
);

/* INSERT QUERY NO: 22 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame22', 0, 1, 1, 1, 0, 0, 0, 1, 'prod_022'
);

/* INSERT QUERY NO: 23 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame23', 1, 1, 1, 1, 1, 1, 1, 0, 'prod_023'
);

/* INSERT QUERY NO: 24 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame24', 1, 1, 1, 1, 1, 1, 0, 1, 'prod_024'
);

/* INSERT QUERY NO: 25 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame25', 1, 1, 1, 1, 1, 1, 1, 1, 'prod_025'
);

/* INSERT QUERY NO: 26 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame26', 0, 1, 1, 1, 1, 1, 0, 0, 'prod_026'
);

/* INSERT QUERY NO: 27 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame27', 1, 1, 1, 1, 1, 0, 0, 0, 'prod_027'
);

/* INSERT QUERY NO: 28 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame28', 0, 1, 1, 1, 1, 0, 0, 0, 'prod_028'
);

/* INSERT QUERY NO: 29 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame29', 1, 1, 1, 1, 1, 1, 0, 0, 'prod_029'
);

/* INSERT QUERY NO: 30 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame30', 1, 1, 1, 1, 1, 0, 0, 1, 'prod_030'
);

/* INSERT QUERY NO: 31 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame31', 1, 1, 1, 1, 0, 0, 0, 0, 'prod_031'
);

/* INSERT QUERY NO: 32 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame32', 1, 1, 1, 0, 1, 0, 0, 1, 'prod_032'
);

/* INSERT QUERY NO: 33 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame33', 1, 1, 1, 1, 1, 0, 0, 1, 'prod_033'
);

/* INSERT QUERY NO: 34 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame34', 1, 1, 1, 1, 1, 1, 0, 1, 'prod_034'
);

/* INSERT QUERY NO: 35 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame35', 1, 1, 1, 1, 0, 0, 0, 1, 'prod_035'
);

/* INSERT QUERY NO: 36 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame36', 1, 0, 1, 1, 0, 0, 0, 0, 'prod_036'
);

/* INSERT QUERY NO: 37 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame37', 1, 1, 1, 1, 1, 0, 0, 0, 'prod_037'
);

/* INSERT QUERY NO: 38 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame38', 1, 0, 1, 1, 0, 0, 0, 1, 'prod_038'
);

/* INSERT QUERY NO: 39 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame39', 1, 1, 1, 1, 1, 0, 0, 0, 'prod_039'
);

/* INSERT QUERY NO: 40 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame40', 1, 0, 1, 1, 1, 1, 0, 1, 'prod_040'
);

/* INSERT QUERY NO: 41 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame41', 1, 1, 1, 1, 1, 1, 0, 1, 'prod_041'
);

/* INSERT QUERY NO: 42 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame42', 1, 1, 1, 1, 1, 1, 1, 0, 'prod_042'
);

/* INSERT QUERY NO: 43 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame43', 1, 1, 1, 0, 1, 0, 0, 0, 'prod_043'
);

/* INSERT QUERY NO: 44 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame44', 1, 0, 1, 1, 0, 0, 0, 0, 'prod_044'
);

/* INSERT QUERY NO: 45 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame45', 1, 1, 0, 1, 1, 0, 1, 0, 'prod_045'
);

/* INSERT QUERY NO: 46 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame46', 1, 0, 1, 1, 1, 0, 0, 1, 'prod_046'
);

/* INSERT QUERY NO: 47 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame47', 1, 1, 1, 1, 0, 1, 0, 0, 'prod_047'
);

/* INSERT QUERY NO: 48 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame48', 1, 1, 1, 1, 1, 0, 0, 0, 'prod_048'
);

/* INSERT QUERY NO: 49 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame49', 1, 1, 1, 1, 1, 1, 1, 1, 'prod_049'
);

/* INSERT QUERY NO: 50 */
INSERT INTO ven_amenities(ven_amenity, ven_ceremony_area, ven_dress_room, ven_handicap_access, ven_indoor, ven_outdoor, ven_liability_ins, ven_covered_outd, ven_accommodation, product_id)
VALUES
(
'ame50', 1, 1, 1, 1, 0, 1, 0, 1, 'prod_050'
);

/* INSERT QUERY NO: 1 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer01', 1, 1, 1, 'prod_001'
);

/* INSERT QUERY NO: 2 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer02', 1, 1, 1, 'prod_002'
);

/* INSERT QUERY NO: 3 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer03', 1, 1, 1, 'prod_003'
);

/* INSERT QUERY NO: 4 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer04', 1, 1, 1, 'prod_004'
);

/* INSERT QUERY NO: 5 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer05', 0, 1, 1, 'prod_005'
);

/* INSERT QUERY NO: 6 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer06', 1, 1, 1, 'prod_006'
);

/* INSERT QUERY NO: 7 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer07', 1, 1, 1, 'prod_007'
);

/* INSERT QUERY NO: 8 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer08', 1, 1, 1, 'prod_008'
);

/* INSERT QUERY NO: 9 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer09', 1, 1, 1, 'prod_009'
);

/* INSERT QUERY NO: 10 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer10', 1, 1, 1, 'prod_010'
);

/* INSERT QUERY NO: 11 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer11', 1, 1, 1, 'prod_011'
);

/* INSERT QUERY NO: 12 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer12', 1, 1, 1, 'prod_012'
);

/* INSERT QUERY NO: 13 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer13', 1, 1, 1, 'prod_013'
);

/* INSERT QUERY NO: 14 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer14', 1, 1, 1, 'prod_014'
);

/* INSERT QUERY NO: 15 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer15', 1, 1, 1, 'prod_015'
);

/* INSERT QUERY NO: 16 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer16', 0, 1, 1, 'prod_016'
);

/* INSERT QUERY NO: 17 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer17', 1, 1, 1, 'prod_017'
);

/* INSERT QUERY NO: 18 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer18', 0, 1, 1, 'prod_018'
);

/* INSERT QUERY NO: 19 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer19', 1, 1, 1, 'prod_019'
);

/* INSERT QUERY NO: 20 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer20', 1, 1, 1, 'prod_020'
);

/* INSERT QUERY NO: 21 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer21', 1, 1, 1, 'prod_021'
);

/* INSERT QUERY NO: 22 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer22', 1, 1, 1, 'prod_022'
);

/* INSERT QUERY NO: 23 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer23', 1, 1, 1, 'prod_023'
);

/* INSERT QUERY NO: 24 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer24', 1, 1, 1, 'prod_024'
);

/* INSERT QUERY NO: 25 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer25', 1, 1, 1, 'prod_025'
);

/* INSERT QUERY NO: 26 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer26', 1, 1, 1, 'prod_026'
);

/* INSERT QUERY NO: 27 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer27', 1, 1, 1, 'prod_027'
);

/* INSERT QUERY NO: 28 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer28', 1, 1, 1, 'prod_028'
);

/* INSERT QUERY NO: 29 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer29', 1, 1, 1, 'prod_029'
);

/* INSERT QUERY NO: 30 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer30', 0, 1, 1, 'prod_030'
);

/* INSERT QUERY NO: 31 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer31', 1, 1, 1, 'prod_031'
);

/* INSERT QUERY NO: 32 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer32', 1, 1, 1, 'prod_032'
);

/* INSERT QUERY NO: 33 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer33', 1, 1, 1, 'prod_033'
);

/* INSERT QUERY NO: 34 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer34', 0, 1, 1, 'prod_034'
);

/* INSERT QUERY NO: 35 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer35', 1, 1, 1, 'prod_035'
);

/* INSERT QUERY NO: 36 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer36', 1, 1, 1, 'prod_036'
);

/* INSERT QUERY NO: 37 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer37', 1, 1, 1, 'prod_037'
);

/* INSERT QUERY NO: 38 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer38', 1, 1, 0, 'prod_038'
);

/* INSERT QUERY NO: 39 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer39', 1, 1, 1, 'prod_039'
);

/* INSERT QUERY NO: 40 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer40', 1, 1, 1, 'prod_040'
);

/* INSERT QUERY NO: 41 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer41', 1, 1, 1, 'prod_041'
);

/* INSERT QUERY NO: 42 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer42', 1, 1, 1, 'prod_042'
);

/* INSERT QUERY NO: 43 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer43', 1, 1, 1, 'prod_043'
);

/* INSERT QUERY NO: 44 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer44', 1, 1, 1, 'prod_044'
);

/* INSERT QUERY NO: 45 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer45', 1, 1, 1, 'prod_045'
);

/* INSERT QUERY NO: 46 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer46', 1, 1, 1, 'prod_046'
);

/* INSERT QUERY NO: 47 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer47', 0, 1, 1, 'prod_047'
);

/* INSERT QUERY NO: 48 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer48', 1, 1, 1, 'prod_048'
);

/* INSERT QUERY NO: 49 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer49', 1, 1, 1, 'prod_049'
);

/* INSERT QUERY NO: 50 */
INSERT INTO ven_ceremonies(ven_ceremony, ven_civil_u, ven_nonrelig_cerem, ven_relig_cerem, product_id)
VALUES
(
'cer50', 1, 1, 0, 'prod_050'
);
```
### Jewelry Department
```
#Data for team 5 Jewelry until 11912
/* CREATE TABLE Material*/
DROP TABLE IF EXISTS material;
CREATE TABLE material(
material_id INT NOT NULL AUTO_INCREMENT,
product_id VARCHAR(255),
material_name VARCHAR(255),
PRIMARY KEY (material_id),
FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

/* CREATE TABLE Category */
DROP TABLE IF EXISTS categories;
CREATE TABLE categories(
category_id INT NOT NULL AUTO_INCREMENT,
product_id VARCHAR(255),
category_name VARCHAR(255),
PRIMARY KEY (category_id),
FOREIGN KEY(product_id)
REFERENCES Products(product_id)
);

/* CREATE TABLE Product Description */
DROP TABLE IF EXISTS product_description;
CREATE TABLE product_description(
product_description_id INT NOT NULL AUTO_INCREMENT,
product_id VARCHAR(255),
product_description VARCHAR(255),
PRIMARY KEY (product_description_id),
FOREIGN KEY (product_id) REFERENCES Products(product_id)
);


/* INSERT QUERY NO: 1 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_084', 'gold'
);

/* INSERT QUERY NO: 2 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_085', 'gold'
);

/* INSERT QUERY NO: 3 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_086', 'gold'
);

/* INSERT QUERY NO: 4 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_087', 'gold'
);

/* INSERT QUERY NO: 5 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_088', 'gold'
);

/* INSERT QUERY NO: 6 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_089', 'gold'
);

/* INSERT QUERY NO: 7 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_090', 'gold'
);

/* INSERT QUERY NO: 8 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_091', 'gold'
);

/* INSERT QUERY NO: 9 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_092', 'silver'
);

/* INSERT QUERY NO: 10 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_093', 'gold'
);

/* INSERT QUERY NO: 11 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_094', 'gold'
);

/* INSERT QUERY NO: 12 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_095', 'gold'
);

/* INSERT QUERY NO: 13 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_096', 'gold'
);

/* INSERT QUERY NO: 14 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_097', 'gold'
);

/* INSERT QUERY NO: 15 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_098', 'gold'
);

/* INSERT QUERY NO: 16 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_099', 'gold'
);

/* INSERT QUERY NO: 17 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_100', 'gold'
);

/* INSERT QUERY NO: 18 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_101', 'gold'
);

/* INSERT QUERY NO: 19 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_102', 'gold'
);

/* INSERT QUERY NO: 20 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_103', 'gold'
);

/* INSERT QUERY NO: 21 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_104', 'gold'
);

/* INSERT QUERY NO: 22 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_105', 'gold'
);

/* INSERT QUERY NO: 23 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_106', 'gold'
);

/* INSERT QUERY NO: 24 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_107', 'gold'
);

/* INSERT QUERY NO: 25 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_108', 'gold'
);

/* INSERT QUERY NO: 26 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_109', 'gold'
);

/* INSERT QUERY NO: 27 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_110', 'gold'
);

/* INSERT QUERY NO: 28 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_111', 'gold'
);

/* INSERT QUERY NO: 29 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_112', 'platinum'
);

/* INSERT QUERY NO: 30 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_113', 'gold'
);

/* INSERT QUERY NO: 31 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_114', 'gold'
);

/* INSERT QUERY NO: 32 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_115', 'gold'
);

/* INSERT QUERY NO: 33 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_116', 'gold'
);

/* INSERT QUERY NO: 34 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_117', 'gold'
);

/* INSERT QUERY NO: 35 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_118', 'gold'
);

/* INSERT QUERY NO: 36 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_119', 'gold'
);

/* INSERT QUERY NO: 37 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_120', 'gold'
);

/* INSERT QUERY NO: 38 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_121', 'gold'
);

/* INSERT QUERY NO: 39 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_122', 'emerald'
);

/* INSERT QUERY NO: 40 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_123', 'sapphire'
);

/* INSERT QUERY NO: 41 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_124', 'brown diamond'
);

/* INSERT QUERY NO: 42 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_125', 'emerald'
);

/* INSERT QUERY NO: 43 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_126', 'emerald'
);

/* INSERT QUERY NO: 44 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_127', 'tanzanite'
);

/* INSERT QUERY NO: 45 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_128', 'sapphire'
);

/* INSERT QUERY NO: 46 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_129', 'ruby'
);

/* INSERT QUERY NO: 47 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_130', 'ruby'
);

/* INSERT QUERY NO: 48 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_131', 'aquamarines'
);

/* INSERT QUERY NO: 49 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_132', 'diamonds'
);

/* INSERT QUERY NO: 50 */
INSERT INTO material(product_id, material_name)
VALUES
(
'prod_133', 'diamonds'
);


/* INSERT Category Name */


/* INSERT QUERY NO: 1 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_084', 'ring'
);

/* INSERT QUERY NO: 2 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_085', 'ring'
);

/* INSERT QUERY NO: 3 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_086', 'necklace'
);

/* INSERT QUERY NO: 4 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_087', 'bracelet'
);

/* INSERT QUERY NO: 5 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_088', 'necklace'
);

/* INSERT QUERY NO: 6 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_089', 'necklace'
);

/* INSERT QUERY NO: 7 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_090', 'necklace'
);

/* INSERT QUERY NO: 8 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_091', 'bracelet'
);

/* INSERT QUERY NO: 9 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_092', 'bracelet'
);

/* INSERT QUERY NO: 10 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_093', 'ring'
);

/* INSERT QUERY NO: 11 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_094', 'bracelet'
);

/* INSERT QUERY NO: 12 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_095', 'ring'
);

/* INSERT QUERY NO: 13 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_096', 'ring'
);

/* INSERT QUERY NO: 14 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_097', 'bracelet'
);

/* INSERT QUERY NO: 15 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_098', 'ring'
);

/* INSERT QUERY NO: 16 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_099', 'ring'
);

/* INSERT QUERY NO: 17 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_100', 'bracelet'
);

/* INSERT QUERY NO: 18 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_101', 'ring'
);

/* INSERT QUERY NO: 19 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_102', 'ring'
);

/* INSERT QUERY NO: 20 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_103', 'ring'
);

/* INSERT QUERY NO: 21 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_104', 'ring'
);

/* INSERT QUERY NO: 22 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_105', 'bracelet'
);

/* INSERT QUERY NO: 23 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_106', 'ring'
);

/* INSERT QUERY NO: 24 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_107', 'ring'
);

/* INSERT QUERY NO: 25 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_108', 'ring'
);

/* INSERT QUERY NO: 26 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_109', 'necklace'
);

/* INSERT QUERY NO: 27 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_110', 'ring'
);

/* INSERT QUERY NO: 28 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_111', 'ring'
);

/* INSERT QUERY NO: 29 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_112', 'ring'
);

/* INSERT QUERY NO: 30 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_113', 'ring'
);

/* INSERT QUERY NO: 31 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_114', 'ring'
);

/* INSERT QUERY NO: 32 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_115', 'ring'
);

/* INSERT QUERY NO: 33 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_116', 'bracelet'
);

/* INSERT QUERY NO: 34 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_117', 'bracelet'
);

/* INSERT QUERY NO: 35 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_118', 'bracelet'
);

/* INSERT QUERY NO: 36 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_119', 'bracelet'
);

/* INSERT QUERY NO: 37 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_120', 'bracelet'
);

/* INSERT QUERY NO: 38 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_121', 'necklace'
);

/* INSERT QUERY NO: 39 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_122', 'necklace'
);

/* INSERT QUERY NO: 40 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_123', 'necklace'
);

/* INSERT QUERY NO: 41 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_124', 'necklace'
);

/* INSERT QUERY NO: 42 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_125', 'necklace'
);

/* INSERT QUERY NO: 43 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_126', 'ring'
);

/* INSERT QUERY NO: 44 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_127', 'ring'
);

/* INSERT QUERY NO: 45 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_128', 'ring'
);

/* INSERT QUERY NO: 46 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_129', 'ring'
);

/* INSERT QUERY NO: 47 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_130', 'bracelet'
);

/* INSERT QUERY NO: 48 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_131', 'bracelet'
);

/* INSERT QUERY NO: 49 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_132', 'bracelet'
);

/* INSERT QUERY NO: 50 */
INSERT INTO categories(product_id, category_name)
VALUES
(
'prod_133', 'bracelet'
);


/* CREATE TABLE Product Description */


/* INSERT QUERY NO: 1 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_084', '14k yellow gold star-studded ring of 0.066 carat total weight'
);

/* INSERT QUERY NO: 2 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_085', '18k yellow ring with sparkling diamonds 0.1 carat total weight'
);

/* INSERT QUERY NO: 3 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_086', '14k yellow gold large necklage '
);

/* INSERT QUERY NO: 4 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_087', '"14k yellow gold star-studded bangle'
);

/* INSERT QUERY NO: 5 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_088', '14k yellow gold small chain necklace '
);

/* INSERT QUERY NO: 6 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_089', '18k yellow gold trio diamond necklace 0.33 carat total weight'
);

/* INSERT QUERY NO: 7 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_090', '14k yellow gold 24-inch rope chain necklace features lustrous gold '
);

/* INSERT QUERY NO: 8 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_091', '14k rose gold 65-inch diamond bangle embellishes the wrist with glimmering diamonds for an elegant and versatile look 0.375 carat total weight'
);

/* INSERT QUERY NO: 9 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_092', '"Silver bracelet glittering cluster comprised of a london blue topaz'
);

/* INSERT QUERY NO: 10 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_093', '14k yellow gold nature-inspired ring features two glistening rows of alternating marquise and round diamond 0.375 carat total weight'
);

/* INSERT QUERY NO: 11 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_094', '14k yellow gold bracelet features five akoya pearls placed evenly across a lustrous cable chain with several sizing stations for increased comfortability breathtaking in hues of white and gold'
);

/* INSERT QUERY NO: 12 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_095', '"14k yellow gold herringbone necklace is as on trend as can be featuring lustrous links that lay smoothly on the skin'
);

/* INSERT QUERY NO: 13 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_096', '"14k rose gold stream of glistening offset diamonds meet at a point in this captivating ring'
);

/* INSERT QUERY NO: 14 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_097', '14k white gold opal diamond gold cuff (carat total weight is not specified)'
);

/* INSERT QUERY NO: 15 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_098', '14k rose gold solitaire claw prong with french pavé eternity band diamond 1.5 carat total weight'
);

/* INSERT QUERY NO: 16 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_099', '14k rose gold dainty diamond stacking ring 0.05 carat total weight'
);

/* INSERT QUERY NO: 17 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_100', '14k rose gold stacking bangles'
);

/* INSERT QUERY NO: 18 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_101', '14k yellow gold emerald cut diamond solitaire ring with east west setting (carat total weight is not specified)'
);

/* INSERT QUERY NO: 19 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_102', '14k yellow gold floating bezel ring in rose gold with black diamond (carat total weight is not specified)'
);

/* INSERT QUERY NO: 20 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_103', '14k white gold signet ring'
);

/* INSERT QUERY NO: 21 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_104', '18k white gold ring with hexagon sapphire 2 carat total weight'
);

/* INSERT QUERY NO: 22 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_105', '14k yellow gold stepped cuff in diamond (carat total weight is not specified)'
);

/* INSERT QUERY NO: 23 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_106', '14k yellow gold champagne oval diamond 1.5 carat total weight engagement ring'
);

/* INSERT QUERY NO: 24 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_107', '14k rose gold montana sapphire engagement ring (carat total weight is not specified)'
);

/* INSERT QUERY NO: 25 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_108', '14k rose gold diamond & pink sapphire rose gold engagement ring (carat total weight is not specified)'
);

/* INSERT QUERY NO: 26 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_109', '14k gold diamond collarbone duster (carat total weight is not specified)'
);

/* INSERT QUERY NO: 27 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_110', '18k rose gold engagement ring with a round center stone and round diamonds 0.80 carat total weight'
);

/* INSERT QUERY NO: 28 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_111', '"18k white gold vintage-inspired ring round cut center stone with marquise diamonds (6.5mm wide'
);

/* INSERT QUERY NO: 29 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_112', 'platinum diamond and sapphire mix dome ring (carat total weight is not specified)'
);

/* INSERT QUERY NO: 30 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_113', '18k white gold ring that contains five hand-selected round brilliant-cut diamonds 0.80 carat total weight '
);

/* INSERT QUERY NO: 31 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_114', '14k white gold  ring with round brilliant-cut diamonds set halfway down the shank (carat total weight is not specified)'
);

/* INSERT QUERY NO: 32 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_115', '18k white gold ring featuring an oval cut center stone with round diamonds set along the shank 0.51 caract total weight'
);

/* INSERT QUERY NO: 33 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_116', '14k white gold tennis bracelet with a total of 202 round diamonds 2.02 carat total weight'
);

/* INSERT QUERY NO: 34 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_117', '"14kyellow gold bracelet with a total of 16 round diamonds'
);

/* INSERT QUERY NO: 35 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_118', '14k yellow gold tennis bracelet with a total of 46 round diamonds 0.62 carat total weight'
);

/* INSERT QUERY NO: 36 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_119', '14k yellow gold bracelet with 14 baguette diamonds approximately 0.40 carat weight and 12 round diamonds approximately 0.22 carat total weight'
);

/* INSERT QUERY NO: 37 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_120', '14k rose gold bracelet with 124 round diamonds 1.11 carat total weight'
);

/* INSERT QUERY NO: 38 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_121', '14k white gold necklace with a double halo featuring 43 round f vs diamonds 0.32 carat total weight'
);

/* INSERT QUERY NO: 39 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_122', '14k withe gold necklace with diamond heart and emerald 0.91 carat total weight'
);

/* INSERT QUERY NO: 40 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_123', '14k white gold necklace with sapphire and diamond 1.00 carat total weigh'
);

/* INSERT QUERY NO: 41 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_124', '14k white gold necklace with brown diamond  rounded off with 14 diamonds around 0.33 carat total weight'
);

/* INSERT QUERY NO: 42 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_125', '14k white gold  with emerald-cut amethyst 4.70 carat total weight'
);

/* INSERT QUERY NO: 43 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_126', '14k white gold ring with london blue topaz with emerald-cut stones around 11.05 carat total weight'
);

/* INSERT QUERY NO: 44 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_127', '14k white gold ring with princess-cut tanzanite and diamond 4.69 carat total weigh'
);

/* INSERT QUERY NO: 45 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_128', '14k white gold ring with solitaire sapphire 0.57 carat total weight'
);

/* INSERT QUERY NO: 46 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_129', '14k white gold ring with ruby 0.57 carat total weigh'
);

/* INSERT QUERY NO: 47 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_130', '14k white gold bracelet 1.08 carat total weight'
);

/* INSERT QUERY NO: 48 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_131', '14k white gold bracelet 3.89 carat total weight'
);

/* INSERT QUERY NO: 49 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_132', '14k white gold chain bracelet with blue diamond 1.00 carat total weight'
);

/* INSERT QUERY NO: 50 */
INSERT INTO product_description(product_id, product_description)
VALUES
(
'prod_133', '14k white gold together beautiful black diamonds in the exquisite bracelet 2.42 carat total weight '
);
```
### Catering Department
```
#Data for team 9 Catering until 12178
/* CREATE TABLE CUISINE */
DROP TABLE IF EXISTS cuisine;
CREATE TABLE cuisine(
product_id VARCHAR(100),
cuisine_id VARCHAR(100),
african VARCHAR(100),
american VARCHAR(100),
asian VARCHAR(100),
bbq VARCHAR(100),
caribbean VARCHAR(100),
french VARCHAR(100),
fusion VARCHAR(100),
german VARCHAR(100),
greek VARCHAR(100),
indian VARCHAR(100),
italian VARCHAR(100),
latin_american VARCHAR(100),
local VARCHAR(100),
mediterranean VARCHAR(100),
mexican VARCHAR(100),
middle_eastern VARCHAR(100),
seafood VARCHAR(100),
southern VARCHAR(100),
spanish VARCHAR(100),
thai VARCHAR(100),
cuisine_product_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
PRIMARY KEY (cuisine_product_id),
FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

/* CREATE TABLE DIETARY*/
DROP TABLE IF EXISTS dietary_option;
CREATE TABLE dietary_option(
product_id VARCHAR(100),
dietary_id VARCHAR(100),
gluten_free VARCHAR(100),
lactose_free VARCHAR(100),
kosher VARCHAR(100),
halal VARCHAR(100),
nut_free VARCHAR(100),
organic VARCHAR(100),
vegan VARCHAR(100),
vegetarian VARCHAR(100),
PRIMARY KEY (dietary_id),
FOREIGN KEY (product_id)  REFERENCES Products(product_id)
);

/* Database for cuisine table */
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_226', 'diet_plated_01', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_227', 'diet_plated_02', '1', '', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_228', 'diet_plated_03', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_229', 'diet_plated_04', '1', '', '', '', '1', '', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_230', 'diet_plated_05', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_231', 'diet_plated_06', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_232', 'diet_plated_07', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_233', 'diet_plated_08', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_234', 'diet_plated_09', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_235', 'diet_plated_10', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_236', 'diet_plated_11', '', '', '', '1', '1', '1', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_237', 'diet_plated_12', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_238', 'diet_plated_13', '1', '', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_239', 'diet_plated_14', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_240', 'diet_plated_15', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_241', 'diet_plated_16', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_242', 'diet_plated_17', '1', '1', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_243', 'diet_plated_18', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_244', 'diet_plated_19', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_245', 'diet_plated_20', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_246', 'diet_plated_21', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_247', 'diet_plated_22', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_248', 'diet_buffet_01', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_249', 'diet_buffet_02', '1', '', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_250', 'diet_buffet_03', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_251', 'diet_buffet_04', '1', '', '', '', '1', '', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_252', 'diet_buffet_05', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_253', 'diet_buffet_06', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_254', 'diet_buffet_07', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_255', 'diet_buffet_08', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_256', 'diet_buffet_09', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_257', 'diet_buffet_10', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_258', 'diet_buffet_11', '', '', '', '1', '1', '1', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_259', 'diet_buffet_12', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_260', 'diet_buffet_13', '1', '', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_261', 'diet_buffet_14', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_262', 'diet_buffet_15', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_263', 'diet_buffet_16', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_264', 'diet_buffet_17', '1', '1', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_265', 'diet_buffet_18', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_266', 'diet_buffet_19', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_267', 'diet_buffet_20', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_268', 'diet_buffet_21', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_269', 'diet_station_01', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_270', 'diet_station_02', '1', '', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_271', 'diet_station_03', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_272', 'diet_station_04', '1', '', '', '', '1', '', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_273', 'diet_station_05', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_274', 'diet_station_06', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_275', 'diet_station_07', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_276', 'diet_station_08', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_277', 'diet_station_09', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_278', 'diet_station_10', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_279', 'diet_station_11', '', '', '', '1', '1', '1', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_280', 'diet_station_12', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_281', 'diet_station_13', '1', '', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_282', 'diet_station_14', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_283', 'diet_station_15', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_284', 'diet_station_16', '1', '1', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_285', 'diet_station_17', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_286', 'diet_station_18', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_287', 'diet_station_19', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_288', 'diet_station_20', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_289', 'diet_starter_01', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_290', 'diet_starter_02', '1', '', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_291', 'diet_starter_03', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_292', 'diet_starter_04', '1', '', '', '', '1', '', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_293', 'diet_starter_05', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_294', 'diet_starter_06', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_295', 'diet_starter_07', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_296', 'diet_starter_08', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_297', 'diet_starter_09', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_298', 'diet_starter_10', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_299', 'diet_starter_11', '', '', '', '1', '1', '1', '', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_300', 'diet_starter_12', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_301', 'diet_starter_13', '1', '', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_302', 'diet_starter_14', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_303', 'diet_starter_15', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_304', 'diet_starter_16', '1', '1', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_305', 'diet_starter_17', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_306', 'diet_starter_18', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_307', 'diet_starter_19', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_308', 'diet_starter_20', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_309', 'diet_starter_21', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_310', 'diet_bar_01', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_311', 'diet_bar_02', '1', '', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_312', 'diet_bar_03', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_313', 'diet_bar_04', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_314', 'diet_bar_05', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_315', 'diet_bar_06', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_316', 'diet_bar_07', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_317', 'diet_bar_08', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_318', 'diet_bar_09', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_319', 'diet_bar_10', '1', '', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_320', 'diet_bar_11', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_321', 'diet_bar_12', '1', '1', '', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_322', 'diet_bar_13', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_323', 'diet_bar_14', '1', '1', '', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_324', 'diet_bar_15', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_325', 'diet_bar_16', '1', '1', '1', '1', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_326', 'diet_bar_17', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_327', 'diet_bar_18', '1', '1', '1', '', '1', '1', '1', '1');
INSERT INTO Dietary_option (product_id, dietary_id, gluten_free, lactose_free, kosher, halal, nut_free, organic, vegan, vegetarian) VALUES ('prod_328', 'diet_bar_19', '1', '1', '', '', '1', '1', '1', '1');





INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_226', 'cus_plated_01', '', '', '', '', '', '', '', '', '', '', '', '', '1', '1', '', '1', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_227', 'cus_plated_02', '', '1', '1', '', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '', '1');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_228', 'cus_plated_03', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_229', 'cus_plated_04', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '1', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_230', 'cus_plated_05', '1', '1', '1', '1', '1', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_231', 'cus_plated_06', '1', '1', '', '1', '1', '1', '', '', '1', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_232', 'cus_plated_07', '', '1', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_233', 'cus_plated_08', '', '1', '', '1', '', '1', '', '', '', '', '1', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_234', 'cus_plated_09', '', '', '', '', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_235', 'cus_plated_10', '', '1', '', '1', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_236', 'cus_plated_11', '', '1', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_237', 'cus_plated_12', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_238', 'cus_plated_13', '', '1', '', '1', '', '', '', '', '', '', '1', '1', '1', '', '', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_239', 'cus_plated_14', '', '1', '', '', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_240', 'cus_plated_15', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_241', 'cus_plated_16', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_242', 'cus_plated_17', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_243', 'cus_plated_18', '1', '1', '', '1', '1', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_244', 'cus_plated_19', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_245', 'cus_plated_20', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_246', 'cus_plated_21', '', '1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_247', 'cus_plated_22', '', '1', '1', '1', '', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_248', 'cus_buffet_01', '', '', '', '', '', '', '', '', '', '', '', '', '1', '1', '', '1', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_249', 'cus_buffet_02', '', '1', '1', '', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '', '1');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_250', 'cus_buffet_03', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_251', 'cus_buffet_04', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '1', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_252', 'cus_buffet_05', '1', '1', '1', '1', '1', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_253', 'cus_buffet_06', '1', '1', '', '1', '1', '1', '', '', '1', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_254', 'cus_buffet_07', '', '1', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_255', 'cus_buffet_08', '', '1', '', '1', '', '1', '', '', '', '', '1', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_256', 'cus_buffet_09', '', '', '', '', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_257', 'cus_buffet_10', '', '1', '', '1', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_258', 'cus_buffet_11', '', '1', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_259', 'cus_buffet_12', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_260', 'cus_buffet_13', '', '1', '', '1', '', '', '', '', '', '', '1', '1', '1', '', '', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_261', 'cus_buffet_14', '', '1', '', '', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_262', 'cus_buffet_15', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_263', 'cus_buffet_16', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_264', 'cus_buffet_17', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_265', 'cus_buffet_18', '1', '1', '', '1', '1', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_266', 'cus_buffet_19', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_267', 'cus_buffet_20', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_268', 'cus_buffet_21', '', '1', '1', '1', '', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_269', 'cus_station_01', '', '', '', '', '', '', '', '', '', '', '', '', '1', '1', '', '1', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_270', 'cus_station_02', '', '1', '1', '', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '', '1');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_271', 'cus_station_03', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_272', 'cus_station_04', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '1', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_273', 'cus_station_05', '1', '1', '1', '1', '1', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_274', 'cus_station_06', '1', '1', '', '1', '1', '1', '', '', '1', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_275', 'cus_station_07', '', '1', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_276', 'cus_station_08', '', '1', '', '1', '', '1', '', '', '', '', '1', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_277', 'cus_station_09', '', '', '', '', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_278', 'cus_station_10', '', '1', '', '1', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_279', 'cus_station_11', '', '1', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_280', 'cus_station_12', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_281', 'cus_station_13', '', '1', '', '1', '', '', '', '', '', '', '1', '1', '1', '', '', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_282', 'cus_station_14', '', '1', '', '', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_283', 'cus_station_15', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_284', 'cus_station_16', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_285', 'cus_station_17', '1', '1', '', '1', '1', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_286', 'cus_station_18', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_287', 'cus_station_19', '', '1', '1', '', '', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_288', 'cus_station_20', '', '1', '1', '1', '', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_289', 'cus_starter_01', '', '', '', '', '', '', '', '', '', '', '', '', '1', '1', '', '1', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_290', 'cus_starter_02', '', '1', '1', '', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '', '1');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_291', 'cus_starter_03', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_292', 'cus_starter_04', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '1', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_293', 'cus_starter_05', '1', '1', '1', '1', '1', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_294', 'cus_starter_06', '1', '1', '', '1', '1', '1', '', '', '1', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_295', 'cus_starter_07', '', '1', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_296', 'cus_starter_08', '', '1', '', '1', '', '1', '', '', '', '', '1', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_297', 'cus_starter_09', '', '', '', '', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_298', 'cus_starter_10', '', '1', '', '1', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_299', 'cus_starter_11', '', '1', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_300', 'cus_starter_12', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_301', 'cus_starter_13', '', '1', '', '1', '', '', '', '', '', '', '1', '1', '1', '', '', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_302', 'cus_starter_14', '', '1', '', '', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_303', 'cus_starter_15', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_304', 'cus_starter_16', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_305', 'cus_starter_17', '1', '1', '', '1', '1', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_306', 'cus_starter_18', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_307', 'cus_starter_19', '', '1', '1', '', '', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_308', 'cus_starter_20', '', '1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_309', 'cus_starter_21', '', '1', '1', '1', '', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_310', 'cus_bar_01', '', '', '', '', '', '', '', '', '', '', '', '', '1', '1', '', '1', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_311', 'cus_bar_02', '', '1', '1', '', '', '1', '1', '', '', '', '', '', '', '', '1', '', '1', '', '', '1');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_312', 'cus_bar_03', '1', '1', '1', '1', '1', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_313', 'cus_bar_04', '1', '1', '', '1', '1', '1', '', '', '1', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_314', 'cus_bar_05', '', '1', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_315', 'cus_bar_06', '', '1', '', '1', '', '1', '', '', '', '', '1', '', '1', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_316', 'cus_bar_07', '', '', '', '', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_317', 'cus_bar_08', '', '1', '', '1', '', '1', '', '', '', '', '1', '1', '1', '1', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_318', 'cus_bar_09', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_319', 'cus_bar_10', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_320', 'cus_bar_11', '', '1', '', '', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_321', 'cus_bar_12', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '1', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_322', 'cus_bar_13', '', '', '', '', '', '', '', '', '', '', '1', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_323', 'cus_bar_14', '', '1', '1', '1', '1', '1', '', '1', '1', '', '1', '1', '1', '1', '1', '', '1', '', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_324', 'cus_bar_15', '1', '1', '', '1', '1', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_325', 'cus_bar_16', '', '1', '1', '1', '1', '1', '', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_326', 'cus_bar_17', '', '1', '1', '', '', '1', '', '', '', '', '1', '1', '1', '1', '1', '', '', '1', '1', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_327', 'cus_bar_18', '', '1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '');
INSERT INTO Cuisine (product_id, cuisine_id, african, american, asian, bbq, caribbean, french, fusion, german, greek, indian, italian, latin_american, local, mediterranean, mexican, middle_eastern, seafood, southern, spanish, thai) VALUES ('prod_328', 'cus_bar_19', '', '1', '1', '1', '', '1', '', '', '1', '', '1', '', '1', '1', '1', '1', '1', '1', '1', '');
```
### Sample Table Output (Venue Department)
Wedding Vendors and Pricing
```
SELECT c.vendor_name, c.vendor_location, c.vendor_rating, a.product_name, a.unit_vol, a.price_unit, a.price_ce
FROM Products AS a
INNER JOIN ven_amenities AS b
ON a.product_id = b.product_id
INNER JOIN vendors AS c
ON a.vendor_id = c.vendor_id
LIMIT 5;
```





