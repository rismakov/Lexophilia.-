# Lexophilia 

## Motivation
The stylometry of written language and the implicit features of a text are able to expose information about the writer to greater detail than purely topic and words used alone. Given this, I examined whether the style of writing is enough to predict the gender of an author. I webscraped articles from multiple different media sites to get a broad source of articles. The articles were all taken from the Political and World sections to examine writing within one topic and prevent potential topic modeling.

## Methods

![Flowchart](/images/flowchart.png)

### Engineered Features
Type token ratio • Mean word length • Mean sentence length • Standard deviation of sentence length • Frequency of commas • Frequency of semicolons • Frequency of exclaimation marks • Frequency of question marks • Polarity • Subjectivity • etc ...

### Differences between Newssites
I compiled data from several different newssites, which included the ones visualized below. To compare, I took the articles of each from within the last two years and compared writing style within that timeframe. As noticable, there were many clear features that differed strongly between them. Complied from last two years: Slate dataset: 145 authors BuzzFeed dataset: 144 authors TIME dataset: 1231 authors TIME_opinion dataset: 556 authors Atlantic dataset: 1647 authors

![Media logos](/images/sites.png)

### Differences between Genders
Strong differences were visible when exploring the data. These were all included in the predictive model. 

## Results

Using Ada Boost, I was able to achieve a higher mean F1 score than both of my baseline models (majority baseline, which predicted the majority class, and a weighed random baseline). This shows that gender is able to be extrapolated from the stylometry of political articles.  

## Web App

You can read about the results to greater degree at: rismakov.com 

Also can play around with the interactive web app that predicts your gender and outputs information on your text: rismakov.com/app 


