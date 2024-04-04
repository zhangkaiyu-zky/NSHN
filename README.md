#IMSI

## Model View
![image](https://github.com/zhangkaiyu-zky/ISMI/blob/main/IMSI/%E6%A8%A1%E5%9E%8B%E5%9B%BE5.png)
## Dataset Specification
- data/ kg_sl2 contains necessary data or scripts for generating data.
  - train.txt: The data included in the training dataset
  - test.txt: The data included in the test dataset
  - valid.txt: The data included in the valid dataset
  - Dataset Details:The KG, denoted as SynLethKG, includes 24 kinds of relationships between 11 entities. Among 24 kinds of relationships, 16 of
them are related to genes directly, e.g. (gene, regulates, gene), (gene,interacts, gene) and (gene, co-varies, gene). And the other 8 relationships are associated with drug and compounds. Besides, 7 out of 11
kinds of entities are directly related to genes, i.e. pathway, cellular
component, biological process, molecular function, disease, compound and anatomy. They are in the format of (gene, relationship,
entity). These entities can be reached from genes in one hop, whereas the other three kinds of entities (pharmacologic class, side effect
  - Dataset link address:http://synlethdb.sist.shanghaitech.edu.cn/v2/#/

- src/ IMSI contains all the source code.
  - modle/: Code for model definition.
  - utils.py: Code for metric calculations and some data preparation.
  - ISMI.py: Code for the functions used in training and evaluation.
  - main.py: Train or evaluate our IMSI Model.


## STEP1:Package Dependency

- Make sure your local environment has the following installed:
  
  - python 3.6+, Tensorflow1.13.1+

## STEP2:Run Model

  ```
  python IMSI.py 
  ```
