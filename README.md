# Search Relevance

### Overview
Search relevance is the measure of accuracy of the relationship between the search query and the search results. Research shows 43% of website visitors go immediately 
to the search bar, and these searchers are about 2-3 times more likely to convert. When users are served results that align to their query and interests, they will be 
more satisfied, more engaged, and even more likely to convert.

### Dataset
Few years back Kaggle hosted a competition, CrowdFlower Search Results Relevance, which had an objective of creating an open-source model that can be used to measure the relevance of search results. In doing so, you'll be helping enable small business owners to match the experience provided by more resource rich competitors. There is approximately 10000 records in the training set, having columns as Query, Product Title, Product Description, Median Relevance, Relevance Variance and there is a test set which contains nearly 22000 records for which we need to set Median Relevance and Relevance Variance.

### EDA
1. There are 261 unique 'query' keyword both in training and test set. Extracted the top 50 most frequently used query string and saw how the train is distributed in that top 50 query keyword for both training and test set and below is their distribution map.

2. There are a few missing 'product_description' both in train and test set. 
Note: We are not imputing the missing value for product description column, as we will be not be using product_description when training our model.

3. The 'median_relevance' field is a catagorical field which contains 4 values, 1 through 4, below graph shows how many number of products belong to each catagory of this relevance score. Apart from this we have a relevance_variance field which is used for measuring the Variance of the relevance scores.

### Feature Engineering 
As part of feature engineering the following steps were performed - 
1. Data cleanup - This includes removing numbers or any punctuation present from the product_title column.
2. Concatenating Data - Concatenate columns 'query' and 'product_title' as a part of feature engineering. This combined data will be feed to our model for training.
 
Examples of combined query and product title - 
<br>'bridal shower decorations Accent Pillow with Heart Design - Red/Black',
<br>'led christmas lights Set of 10 Battery Operated Multi LED Train Christmas Lights - Clear Wire',

4. TFIDF - TFIDF is used for converting the combined text to numerical value, and the ngram parameter of TF-IDF was set to 1 through 5 so it considers creating numerical reference for combination of words within one quey+product_title_pair. Below is an example of how the TF-IDF for query - **bridal shower decorations**

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

### Performance
