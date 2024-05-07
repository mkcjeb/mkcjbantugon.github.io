---
title: "Air France Marketing: R Analysis"
date: 2023-10-08
tags: [SQL, bin method, challenge]
header:
  image: "/images/air france.jpeg"
excerpt: "(SQL) Choose one film based on alternating numeric and non-numeric yes/no questions. Only 3 maximum questions per column are allowed. "
mathjax: "true"
toc: true
toc_label : "Navigate"
---
By: Team Matrix <br>
Michelle Kae Celine Jo-anne Bantugon<br>
Srinivas Kondeti <br>
Zamambo Mkhize <br>
Miron Tkachuk<br>

SQL Challenge built by Professor Chase Kusterer <br>
Hult International Business School<br>

For the initial five questions, we use the "binary search" methodology. Our approach involves initially narrowing down the set of possible options by half. Then, we consider the movie titles starting with vowel letters. Lastly, we check if the duration is longer than X minutes, which depends on the results.

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

<b> Results </b><br>
![SQL Query Results](/images/sql_movie_part_1.png)
![SQL Query Results](/images/sql_movie_part_2.png)
