Twitter sentiment analysis
==

Introduction
--
Social media today has become a very popular communication tool among Internet users. Millions of messages are appearing daily in popular web-sites that provide services such as Twitter, Tumblr, Facebook. Authors of those messages write about their life, share opinions on variety of topics and discuss current issues. As more and more users post about products and services they use, or express their political and religious views, social media web-sites become valuable sources of people’s opinions and sentiments. 
It is observed that some people misuse it to tweet hateful content. These contents will affect the peace and stability of some regions and disrupt the user experience. If we can use machine learning algorithms to automatically categorize the content posted by users, then the platform can automatically filter these comments based on the results of the categorization.

In order to initially achieve the above-mentioned purpose. We decided to use the Twitter sentiment corpus, using different feature sets and machine learning classifiers to determine the best combination for Twitter sentiment analysis. We also experimented with various pre-processing steps like - punctuations, emoticons, twitter specific terms and stemming. We investigated the following features - unigrams, bigrams, trigrams and negation detection. We finally train our classifier using various machine-learning algorithms -  Linear Regression , Decision Trees, Random Forest and Naive Bayes

2- Q：Comment on the important parts of your code <br>
A：We have added comments to the code block

3- Q：Show the parts of the code that you wrote and the parts that you imported from other resources. For instance, you can show that by the comments  <br>
A：We read a lot of literature and referenced some previous experience in preprocessing and models. We also refer to the experience and code implementation of some similar projects, and most of the code is completed by ourselves.

4- Q: Show whether you have done any data preprocessing <br>
A: The next step will show this

#### References
https://pdfs.semanticscholar.org/ad8a/7f620a57478ff70045f97abc7aec9687ccbd.pdf<br>
http://www.aclweb.org/website/old_anthology/S/S13/S13-2.pdf#page=526<br>
http://oro.open.ac.uk/34929/1/76490497.pdf

Data Preprocess
--
#### Dataset
We use the Twitter sentiment corpus provided by Kaggle. Each entry in the corpus contains Tweet id, Topic and Sentiment tags.

#### Preprocessing
Unlike other training data, the Twitter text dataset does not follow certain patterns. Users do not write according to a uniform rule, so it is necessary to preprocess the data set.

1. Replace all non-English characters such as punctuation, numbers, etc. with spaces
2. Replace all uppercase characters with lowercase
3. Lemmatization all the word，like “ate” ’eat’
4. remove all English stop words like[['i', 'me', 'my', 'myself’, 'don', 'should', 'now', 'd',]

use_idf means Term Frequency Inverse Document Frequency (TF-IDF) is a feature vectorization method used in text mining to reflect the importance of terms to documents in a corpus. For example: "a", "the" and "of". If a term occurs frequently throughout the corpus, it means that the term does not contain special information about a particular document. This causes frequently occurring words to be weighted higher than those that are less frequent but more meaningful.

Norm=‘l2’ means normalization. It can reduce the impact of dimensions and speed up program execution.


Exploratory Data Analysis
--
Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

This is the basic data fan chart of our project
![](https://www.kaggleusercontent.com/kf/94595256/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..6F1jIf3M8L8vY1JDNLtLeA.0CogMIRUaxnWHXYXF3S2uMH9o8iC41MNVw3HYFmJwBQyFnOO-OGNazRlYHPyYbOBrs32QZAKkLrvbClU8HA9NTCA_LyhPA7Uv4KtQoez1zTS6SZhGyG_z33fUkvD99vhyUP30bJz7arLRi79UhBQgOKeAMW45ZMm-3v-eYk_XnD_H8t0YTq8IO5JYzuNo38FNPv-rgPFxEpzhU9s7OtYcdAf4Sj8-jz-INXOMtmVcpezQKcsoNZ3mMN5E1EwC_qzwkzb6iGNXoKoSvBSTbdNlUQMuEsNY6v96cc4V_YiNk6nb9CpyZMxCTxojFHBEEWP8llit7nSpXtneDjhOKFVICdNq53sEGy-QF3w4LSqDEpA0aBsGLtnyjjVNzvKncKEp8gyp-9PzkgDtQKbvO9S_3567aZsOoTP9k7Ju8v3bg4avRsTXw0bRsI3MC-YM0uBoJAVYadK4BTYKNe-uJVI3lp4T_vWndN4rDWqyzBEgFHjFT1TatcWgf8nqksqH04-G_LkYIdoSqi9e8fX0o87UAYY3GYmLIUTIokf_xRVtSQS3QR47Ws9VFgN43nJ6s_jPah7r-O3As_PiS07VlplP8Fz9VmcvqmVblai3DEe7PMJ7S_8FVvTyaXoPSk_4rrYPnSvvRstqQ-OOOwsjnrOGa3Tjrz5UgaXMpqtw1WxKEE.AbmyIyh_cJBUGDMgk5RJFA/__results___files/__results___43_1.png)

Analysis
--
1.Linear regression
```python
# Building Logistic Regression Classifier

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
pred = log_reg_model.predict(X_test)
pred_prob = log_reg_model.predict_proba(X_test)
classification_summary(pred,pred_prob,'Logistic Regression (LR)')
```
2.Decison Tree
```python
# Building Decision Tree Classifier

DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
pred = DT_model.predict(X_test)
pred_prob = DT_model.predict_proba(X_test)
classification_summary(pred,pred_prob,'Decision Tree Classifier (DT)')
```

3. Random Forest
```python
# Building Random Forest Classifier

RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
pred = RF_model.predict(X_test)
pred_prob = RF_model.predict_proba(X_test)
classification_summary(pred,pred_prob,'Random Forest Classifier (RF)')
```
4. Naive Bayes
```python
# Building Naive Bayes Classifier

NB_model = BernoulliNB()
NB_model.fit(X_train,y_train)
pred = NB_model.predict(X_test)
pred_prob = NB_model.predict_proba(X_test)
classification_summary(pred,pred_prob,'Naïve Bayes Classifier (NB)')
```


Conclusion
--
From the above experimentation we can conclude that:
  ·Basic preprocessing techniques help us to get rid of unwanted character and gave us the clean data.

  ·The labels in the target variable were somewhat uniformally distributed.

  ·The performace of models were almost similar.

  ·Considering the all metrics, we see Random Forest Classifier performed the best on the current dataset.

  ·Being an equal contendor, it is wise to also consider simpler models like Logisitic Regression as it is more generalisable & computationally less expensive.



