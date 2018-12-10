## Tagging evaluation

##### My approach
I started the program by trying to extract all possible information which we already know from the training corpus. As a result, I use regex to extract speakers and locations by their tags from the training emails. This information is then stored for later use.
After that, I iterate over each email in the untagged directory and split each email into the header and body. I found that many headers contain a lot of the key information which we need to tag such as the speakers, location and times. Hence I try to extract these explicitly from the header and possibly the body if this fails. This data is then stored so that I can tag it in the email as a whole. In tagging paragraphs and sentences, I use regex to identify the paragraphs and sentences in the text and place the appropriate tags in their respective place. This worked a lot of the time however sometimes I found that it struggled with text that was formatted in a non-standard format e.g. paragraphs hanging in the middle of the email not anchored to the margins well. However, I understand that in natural language processing perfection is not expected. After this, I join the header and body together and search for all occurrences of the data I extracted such as times, locations and speakers and tag these; subsequently writing the fully tagged email to a directory of output files.
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

Although the approach I use has obtained quite good results I feel that given a potentially larger corpus, it would have been interesting to attempt a solution which uses machine learning to try and train on the training set and tag the untagged emails. I feel like this would allow the classifier to develop an understanding of under which contexts we typically see a speaker, time or location introduced and may obtain better results than my more standard approach, and given more time, I would attempt this.

## Ontology classification evaluation
I was able to achieve fairly reliable results in ontology classification through the use of GloVe. My approach to this was as follows:
* Define a set of keywords for certain categories
* Convert vector file into dictionary of <word, vector> pairs
* Obtain topic from an email
* Remove stop words and non-nouns
* Use nouns in the topic to calculate how likely each word in a topic is likely to occur next in the same context as other keywords for each category
* Use this to generate a score for each category
* If it is not possible to classify using the topic then count the number of keywords which occur in the body
* Classify body as the topic with most occurrences of keywords
#### *Advantages*
The advantages of this approach are that it always tries to classify the text and normally classifies well due to the set of related keywords.
#### *Disadvantages*
The disadvantages of this method are that for it to work well you need to construct a large set of keywords with little overlap for it to work well. Not only that but ignoring nouns means that occasionally, some important words such as some adjectives which help describe the context of the noun are thrown away and not used to improve the classification of the text.
    To help combat this, if a word in the topic is a keyword in any category, I boost the score for that category for that word.


