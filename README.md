# Implicit User Awareness Modeling via Candidate Items for CTR Prediction in Search Ads

This repo is the official implementation for the [SIGIR 2023 paper: *LOVF: Layered Organic View Fusion for Click-through Rate Prediction in Online Advertising*].

## Data format

Our released dataset *ORAD-Pub* could be found at [JD JingPan](http://box.jd.com/sharedInfo/36F44FFB7B3AEC1CEEC946AFBC5707A7) with password `3jhbd6`.

In the data files, each row corresponds to a search session. 
Each column is a piece of multiple sample data aggregated according to user-query. The organization form of each column is:
column[0]: user_id. int type
column[1]: query_id. int type
column[2]: The source of each sample(0:advertising. 1:organic). list type
column[3]: The label of each sample. list type
column[4:]: the side info [itemID, categoryID, brandID, vendorID and priceID] of each sample. list type

## Requirements

* python 3.6.13
* tensorflow 1.15.0
* scikit-learn 0.24.2


## Quick start

Create a new `data` folder and put the downloaded dataset into the folder. Then,

```bash
python src/main.py 
```
