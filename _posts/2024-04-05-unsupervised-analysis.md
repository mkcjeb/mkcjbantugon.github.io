---
title: "Facebook Data: Unsupervised Analysis (Coming Soon)"
date: 2024-04-14
tags: [Python, machine learning, unsupervised, cluster, k-means, PCA, classification]
header:
  image: "/images/fb live.png"
excerpt: "(Python - Machine Learning) This is to analyze the impact of photo content. It employs various machine learning techniques such as Logistic Regression, Principal Component Analysis (PCA), and k-means clustering to analyze the variability of consumer engagement. By understanding the effectiveness of different types of content based on engagement metrics, this analysis aims to provide valuable insights to marketing firms and social media consultants. These insights can help optimize content creation strategies and increase engagement on Facebook in Thailand's dynamic social media environment. "
mathjax: "true"
toc: true
toc_label : "Navigate"
---
By: Michelle Kae Celine Jo-anne Bantugon<br>

Business case built by Professor Chase Kusterer<br>
Hult International Business School<br>

Jupyter notebook, dataset, and factor loading for this analysis can be found here: [Data Science Portfolio](https://github.com/mkcjeb/data-science-portfolio/tree/main/projects/unsupervised_facebook_live_data) 

### Introduction
As digitalization and the use of social media continue to increase, platforms like Facebook have become vital for small vendors to expand their reach and engage with a larger audience especially in Thailand. This transformation has significantly influenced marketing strategies, particularly in how individuals interact with Facebook content. Engagement refers to various actions users take on posts, pages, groups, or ads, including comments, shares, likes, and reactions like love, wow, haha, angry, and sad. This analysis focuses on studying the impact of photo content on Facebook engagement in Thailand. It employs various machine learning techniques such as Logistic Regression, Principal Component Analysis (PCA), and k-means clustering to analyze the variability of consumer engagement. By understanding the effectiveness of different types of content based on engagement metrics, this analysis aims to provide valuable insights to marketing firms and social media consultants. These insights can help optimize content creation strategies and increase engagement on Facebook in Thailand's dynamic social media environment.

### Conclusion
The analysis showed valuable insights into the impact of different types of content, particularly photos, on Facebook engagement in Thailand. Across various clusters, photos come out as a popular and effective means of communication, prompting positive reactions such as likes and wows as engagement. Moreover, the study also highlights the importance of diversifying content strategies to cater to different audience preferences. While some clusters demonstrate a strong preference for photo posts, others show greater engagement with video or text content.

### Recommendation
<b> 1. Content and Relationship Strategy </b><br>
Photo Emphasis <br>
With the popularity of photos, company should focus on creating visually appealing and engaging photo content. Use high-quality images that are likely to stimulate positive emotions and encourage likes.<br>
Diversification <br>
While photos are popular, consider diversifying content with videos and text posts to cater to different audience preferences. Cluster 3, for example, shows a strong interest in videos, suggesting an opportunity to explore video content creation.<br>
Emotional Appeal <br>
Ensure that the content create positive emotions such as happiness, excitement, or inspiration, as these are more likely to result in likes and positive engagement. This is aligned with the findings by Lee et al. (2016) who found out that enjoyment of content and interpersonal relationships were the two most common motives for liking. <br>

<b> 2. Engagement Strategies </b><br>
Interactive Content <br>
Create engaging and interactive content that encourages comments, shares, and other forms of reactions. The use of unusual, participatory activities such as contests, games, and artistic performances will drive reactions and comments (Apiradee Wongkitrungrueng, Nassim Dehouche & Nuttapol Assarut, 2020). <br>
Storytelling <br>
Use storytelling techniques to make your content more relatable and engaging. People tend to engage more with content that resonates with them on a personal level. <br>

<b> 3. Track Performance </b><br>
Monitor and evaluate photo content performance metrics and other status types on a regular basis to find out what appeals to the audience. Utilize analytical tools to monitor engagement metrics such as likes, shares, comments, and others. <br>

### Part I. 
