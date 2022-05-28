Twitter sentiment analysis
==

Introduction
--
Social media today has become a very popular communication tool among Internet users. Millions of messages are appearing daily in popular web-sites that provide services such as Twitter, Tumblr, Facebook. Authors of those messages write about their life, share opinions on variety of topics and discuss current issues. As more and more users post about products and services they use, or express their political and religious views, social media web-sites become valuable sources of people’s opinions and sentiments. 
It is observed that some people misuse it to tweet hateful content. These contents will affect the peace and stability of some regions and disrupt the user experience. If we can use machine learning algorithms to automatically categorize the content posted by users, then the platform can automatically filter these comments based on the results of the categorization.

In order to initially achieve the above-mentioned purpose. We decided to use the Twitter sentiment corpus, using different feature sets and machine learning classifiers to determine the best combination for Twitter sentiment analysis. We also experimented with various pre-processing steps like - punctuations, emoticons, twitter specific terms and stemming. We investigated the following features - unigrams, bigrams, trigrams and negation detection. We finally train our classifier using various machine-learning algorithms -  Linear Regression , Decision Trees, Random Forest and Naive Bayes


Data Preprocess
--
####Dataset
We use the Twitter sentiment corpus provided by Kaggle. Each entry in the corpus contains Tweet id, Topic and Sentiment tags.

####Preprocessing
Unlike other training data, the Twitter text dataset does not follow certain patterns. Users do not write according to a uniform rule, so it is necessary to preprocess the data set.

1. Replace all non-English characters such as punctuation, numbers, etc. with spaces
2. Replace all uppercase characters with lowercase
3. Lemmatization all the word，like “ate” ’eat’
4. remove all English stop words like[['i', 'me', 'my', 'myself’, 'don', 'should', 'now', 'd',]

use_idf means Term Frequency Inverse Document Frequency (TF-IDF) is a feature vectorization method used in text mining to reflect the importance of terms to documents in a corpus. For example: "a", "the" and "of". If a term occurs frequently throughout the corpus, it means that the term does not contain special information about a particular document. This causes frequently occurring words to be weighted higher than those that are less frequent but more meaningful.

Norm=‘l2’ means normalization. It can reduce the impact of dimensions and speed up program execution.


Exploratory Data Analysis
--
![](https://www.kaggleusercontent.com/kf/94595256/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..6F1jIf3M8L8vY1JDNLtLeA.0CogMIRUaxnWHXYXF3S2uMH9o8iC41MNVw3HYFmJwBQyFnOO-OGNazRlYHPyYbOBrs32QZAKkLrvbClU8HA9NTCA_LyhPA7Uv4KtQoez1zTS6SZhGyG_z33fUkvD99vhyUP30bJz7arLRi79UhBQgOKeAMW45ZMm-3v-eYk_XnD_H8t0YTq8IO5JYzuNo38FNPv-rgPFxEpzhU9s7OtYcdAf4Sj8-jz-INXOMtmVcpezQKcsoNZ3mMN5E1EwC_qzwkzb6iGNXoKoSvBSTbdNlUQMuEsNY6v96cc4V_YiNk6nb9CpyZMxCTxojFHBEEWP8llit7nSpXtneDjhOKFVICdNq53sEGy-QF3w4LSqDEpA0aBsGLtnyjjVNzvKncKEp8gyp-9PzkgDtQKbvO9S_3567aZsOoTP9k7Ju8v3bg4avRsTXw0bRsI3MC-YM0uBoJAVYadK4BTYKNe-uJVI3lp4T_vWndN4rDWqyzBEgFHjFT1TatcWgf8nqksqH04-G_LkYIdoSqi9e8fX0o87UAYY3GYmLIUTIokf_xRVtSQS3QR47Ws9VFgN43nJ6s_jPah7r-O3As_PiS07VlplP8Fz9VmcvqmVblai3DEe7PMJ7S_8FVvTyaXoPSk_4rrYPnSvvRstqQ-OOOwsjnrOGa3Tjrz5UgaXMpqtw1WxKEE.AbmyIyh_cJBUGDMgk5RJFA/__results___files/__results___43_1.png)





