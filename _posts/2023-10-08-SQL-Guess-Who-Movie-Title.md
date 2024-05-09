---
title: "Post: Gallery"
categories:
  - Post Formats
tags:
  - gallery
  - Post Formats
  - tiled
gallery:
  - url: /images/The_Matrix_SQL_Flowchart.jpg
    image_path: /images/The_Matrix_SQL_Flowchart.jpg
    alt: "placeholder image 1"
    title: "Image 1 title caption"
  - url: /images/sql_movie_part_1.png
    image_path: /images/sql_movie_part_1.png
    alt: "placeholder image 2"
    title: "Image 2 title caption"
  - url: /images/sql_movie_part_2.png
    image_path: /images/sql_movie_part_2.png
    alt: "placeholder image 3"
    title: "Image 3 title caption"
  - url: /images/sql_movie_part_1.png
    image_path: /images/sql_movie_part_1.png
    alt: "placeholder image 4"
    title: "Image 4 title caption"
---
By: Team Matrix <br>
Michelle Kae Celine Jo-anne Bantugon<br>
Srinivas Kondeti <br>
Zamambo Mkhize <br>
Miron Tkachuk<br>

SQL Challenge built by Professor Chase Kusterer <br>
Hult International Business School<br>

For the initial five questions, we use the "binary search" methodology. Our approach involves initially narrowing down the set of possible options by half. Then, we consider the movie titles starting with vowel letters. Lastly, we check if the duration is longer than X minutes, which depends on the results.

### SQL Query
```
SELECT *, 
       id % 2 = 1 AS id_odd,                 --  Is ID ODD number?
       LENGTH(title) % 2 = 1 AS title_odd,   --  Is length of the title an ODD number?
       release_year % 2 = 1 AS ryear_odd,    -- Is the release year an ODD number?
       certification LIKE '_' AS cert_one,   -- Does the certification consist of one letter?
       duration % 2 = 1 AS duration_odd      -- Is the duration an ODD number?
FROM films
WHERE id IS NOT NULL AND
      title IS NOT NULL AND
      release_year IS NOT NULL AND
      country IS NOT NULL AND
      duration IS NOT NULL AND
      language IS NOT NULL AND
      certification IS NOT NULL AND
      gross IS NOT NULL AND
      budget IS NOT NULL
GROUP BY id_odd, title_odd, ryear_odd, cert_one, duration_odd;
```
### Flowchart
![Flowchart](/images/The_Matrix_SQL_Flowchart.jpg)

### <b> Results </b><br>
![SQL Query Results](/images/sql_movie_part_1.png)
<br>

```yaml
gallery:
  - url: /images/The_Matrix_SQL_Flowchart.jpg
    image_path: /images/The_Matrix_SQL_Flowchart.jpg
    alt: "placeholder image 1"
    title: "Image 1 title caption"
  - url: /images/sql_movie_part_1.png
    image_path: /images/sql_movie_part_1.png
    alt: "placeholder image 2"
    title: "Image 2 title caption"
  - url: /images/sql_movie_part_2.png
    image_path: /images/sql_movie_part_2.png
    alt: "placeholder image 3"
    title: "Image 3 title caption"
  - url: /images/sql_movie_part_1.png
    image_path: /images/sql_movie_part_1.png
    alt: "placeholder image 4"
    title: "Image 4 title caption"
```

![SQL Query Results](/images/sql_movie_part_2.png)
