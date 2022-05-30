# Search Relevance

### Overview
Search relevance is the measure of accuracy of the relationship between the search query and the search results. Research shows 43% of website visitors go immediately 
to the search bar, and these searchers are about 2-3 times more likely to convert. When users are served results that align to their query and interests, they will be 
more satisfied, more engaged, and even more likely to convert.

### Dataset
Few years back Kaggle hosted a competition, CrowdFlower Search Results Relevance, which had an objective of creating an open-source model that can be used to measure the relevance of search results. In doing so, you'll be helping enable small business owners to match the experience provided by more resource rich competitors. There is approximately 10000 records in the training set, having columns as Query, Product Title, Product Description, Median Relevance, Relevance Variance and there is a test set which contains nearly 22000 records for which we need to set Median Relevance and Relevance Variance.

### EDA
1. There are 261 unique 'query' keyword both in training and test set. Extracted the top 50 most frequently used query string and saw how the train is distributed in that top 50 query keyword for both training and test set and below is their distribution map.

<img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/Top50.png'>

2. There are a few missing 'product_description' both in train and test set. 
Note: We are not imputing the missing value for product description column, as we will be not be using product_description when training our model.

<img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/null_pct.png'>

3. The 'median_relevance' field is a catagorical field which contains 4 values, 1 through 4, below graph shows how many number of products belong to each catagory of this relevance score. Apart from this we have a relevance_variance field which is used for measuring the Variance of the relevance scores.

<br> Median Variance Distribution -
<br><img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/med_rel.png'>

<br> Relevance Variance
<br><p float="left">Query - **Wireless Mouse** <br>
 <img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/rel_var_wmouse.png' width=500 height=200><img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/rel_var_wmouse1.png' width=500 height=200>
</p>
<br><p float="left">Query - **Bike LocK** <br>
 <img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/rel_var_bike.png' width=500 height=200><img src='https://github.com/Ruparna25/search_relevance/blob/main/Images/rel_var_bike1.png' width=500 height=200>
</p>

### Feature Engineering 
As part of feature engineering the following steps were performed - 
1. Data cleanup - This includes removing numbers or any punctuation present from the product_title column.
2. Concatenating Data - Concatenate columns 'query' and 'product_title' as a part of feature engineering. This combined data will be feed to our model for training.
 
Examples of combined query and product title - 
<br>'bridal shower decorations Accent Pillow with Heart Design - Red/Black',
<br>'led christmas lights Set of 10 Battery Operated Multi LED Train Christmas Lights - Clear Wire',

3. TFIDF - TFIDF is used for converting the combined text to numerical value, and the ngram parameter of TF-IDF was set to 1 through 5 so it considers creating numerical reference for combination of words within one quey+product_title_pair. Below is an example of how the TF-IDF for query - **bridal shower decorations**

 **5-GRAM**
<br>('bridal shower decorations accent pillow', 'b') 0.261742125188547

**4-GRAM**
<br>('shower decorations accent pillow', 'b') 0.261742125188547  
<br>('bridal shower decorations accent', 'b') 0.261742125188547 

**3-GRAM**
<br>('decorations accent pillow', 'b') 0.261742125188547 
<br>('shower decorations accent', 'b') 0.261742125188547
<br>('bridal shower decorations', 'b') 0.1956813789955167 

**2-GRAM**
<br>('red black', 'b') 0.2377917675212798
<br>('heart design', 'b') 0.2685203837650798
<br>('accent pillow', 'b') 0.261742125188547
<br>('decorations accent', 'b') 0.261742125188547
<br>('shower decorations', 'b') 0.1956813789955167
<br>('bridal shower', 'b') 0.1956813789955167

**1-GRAM**
('black', 'b') 0.09900061546307047
('red', 'b') 0.14414965478495423
('design', 'b') 0.18429960671355355
('heart', 'b') 0.1811466520220755
('pillow', 'b') 0.1557160855343555
('accent', 'b') 0.2083450387130452
('decorations', 'b') 0.194998738484414
('shower', 'b') 0.1900282943976819
('bridal', 'b') 0.19433110228300043

### Model
Creating Pipeline - 
1. After the TFIDF transformation of the data, a data pipeline is created. The first operation performed as a part of the data pipeline is to do dimensionality reduction of the transformed data. As TF-IDF creates a sparse matrix so we need to decompose the transformed data for ease of training. TruncatedSVD class of sklearn.preprocessing is used to reduce the dimension of the training data.

2. The truncated data is then standardized using StandardScaler class of sklearn.preprocesing. This is done for normalizing the data.

3. The third operation of the pipeline is modeling. The transformed and normalized data is fed to a model for training. The model I tried here is SVC - Support Vector Classifier which accepts the preprocessed data as feature and median_relevance as the target variable. 

Model Training -
<br> Used GridSearchCV for hyperparamenter tuning. The comibination of parameters used for training the model were - Reduced Dimension - 200/400 and 'C' parameter of SVC was set to 10/12 and cross_validation size was set to 2. Based on these figures, there were total of 8 fits of the model. The best model was obtained with the below parameter combination - 

Pipeline(steps=[('svd', TruncatedSVD(n_components=200)),
                ('std', StandardScaler()), ('svm', SVC(C=10))])

### Performance
The metrics used for scoring the model performance is **kappa statistics**. Below is a brief description of what kappa statistics refer to -
<br>The kappa statistic is frequently used to test interrater reliability. The importance of rater reliability lies in the fact that it represents the extent to which the data collected in the study are correct representations of the variables measured. Measurement of the extent to which data collectors (raters) assign the same score to the same variable is called interrater reliability. While there have been a variety of methods to measure interrater reliability, traditionally it was 
measured as percent agreement, calculated as the number of agreement scores divided by the total number of scores. 
