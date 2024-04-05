#NSHN

## Model View
![image](https://github.com/zhangkaiyu-zky/NSHN/blob/main/1.png)
## Dataset Specification
- dataset/ Douban contains necessary data or scripts for generating data.
  - train.txt: The data included in the training dataset
  - test.txt: The data included in the test dataset
  
  - Dataset Details:Douban-Book dataset is a dataset containing book information on the Douban website. This data set usually includes basic information about the book, such as title, author, publication date, ISBN, etc., as well as user ratings, comments and other information on the book.
 
  - dataset/ gowalla contains necessary data or scripts for generating data.
  - train.txt: The data included in the training dataset
  - test.txt: The data included in the test dataset
  
  - Dataset Details:The Gowalla dataset is a historically popular location dataset that contains user check-in data collected by the Gowalla social networking application. This data set records information about users checking in at different locations (such as restaurants, attractions, shops, etc.), as well as check-in timestamps and other related information.
  - 
 - dataset/ yelp2018 contains necessary data or scripts for generating data.
  - train.txt: The data included in the training dataset
  - test.txt: The data included in the test dataset
    - Dataset Details:The Yelp 2018 dataset is a dataset provided by Yelp that contains business information, user reviews, and user behavior data. Yelp is a widely used user review and business information sharing platform where users can find business information, view other users' reviews, and comment and rate.

- NSHN contains all the source code.
  - modle/: Code for model definition.
  - utils.py: Code for metric calculations and some data preparation.
  - main.py: Train or evaluate our NSHN Model.


## STEP1:Package Dependency

- Make sure your local environment has the following installed:
  
  - python 3.6+, pytorch.13.1+
  - tensorflow==1.14.0
  - scipy==1.6.2
  - numpy==1.20.3

## STEP2:Run Model

  ```
  python main.py 
  ```

## STEP3:Settings of key hyperparameters
We observed the change in model efficiency by assigning
different values to the denoising threshold, and the threshold
𝜏 affects the loss rate in the denoising method. To analyze
the effect of 𝜏, we vary the value of 𝜏 in the range of 0-
0.5 and show the results in the figure. Specifically, when 𝜏
= 0.1, 
