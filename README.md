## Tagging evaluation


The following table shows a statistical evaluation of my code to tag the seminar emails provided:
```
Tag          Precision    Recall    F Measure
---------  -----------  --------  -----------
stime         0.935657  0.913613     0.924503
etime         0.957831  0.868852     0.911175
location      0.546099  0.57037      0.557971
speaker       0.461929  0.540059     0.497948
sentence      0.878862  0.731889     0.79867
paragraph     0.712291  0.546039     0.618182
```

## Ontology classification evaluation
I was able to achieve fairly reliable results in ontology classification through the use of GloVe. My approach to this was as follows:
* Define a set of keywords for certain categories
* Convert vector file into dictionary of <word, vector> pairs
* Obtain topic from an email
* Remove stop words and non-nouns
* Use nouns in the topic to calculate how likely each word in a topic is likely to occur next in the same context as other keywirds for each category
* Use this to generate a score for each category
* If it is not possible to classify using the topic then count the number of keywords which occur in the body
* Classify body as topic with most occurences of keywords
#### *Advantages*
The advantages of this approach are that it always tries to classify the text and normally classifies well due to the set of related keywords.
#### *Disadvantages*
The disadvatges of this method are that for it to work well you need to construct a large set of keywords with little overlap for it to work well. Not only that but ignoring nouns means that occasionally, some important words such as some adjectives which help descrive the context of the noun are thrown away and not used to improve the classification of the text.
    To help combat this, if a word in the topic is a keyword in any category, I boost the score for that category for that word.


