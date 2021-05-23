# Open Data Exploratory Analysis

Corresponding GitHub repository for the paper: ''


Maintained by: dadler@infosci.cornell.edu (GitHub ID: dadler6)

## Results



## Data

The cleaned data used for the lasso regression analysis can be found in the `data` directory. The cleaned datasets are the CrossCheck and StudentLife datasets, with epoch features computed, and these features are averaged across 3 days to align with the EMAs. The cleaned data is called:

* Cleaned CrossCheck: `data/crosscheck\_daily\_data\_cleaned\_w\_sameday.csv`
* Cleaned StudentLife: `data/studentlife\_daily\_data\_cleaned\_w\_sameday\_03192020.csv`

These files are used in the `notebooks/lasso\_regression.ipynb` notebook.

Due to filesize, we are not including the original public datasets in this repository, but they are needed if you wish to run the cleaning code (`notebooks/data\_cleaning.ipynb`). Please download them at the following links:

* Raw CrossCheck Data is the `CrossCheck\_Daily\_Data.csv` on Box: https://cornell.box.com/s/rkx46bgv36lkmo2eu349ka95senn48gh
* Raw StudentLife Data must be downloaded at unzipped from: https://studentlife.cs.dartmouth.edu/dataset.html

## Code

The code to run the Lasso Regression models can be found in the following notebook:

* `notebooks/lasso\_regression.ipynb`

The code to clean the original public datasets can be found in:

* `notebooks/data\_cleaning.ipynb`

### Utility code

The following `.py` files are used throughout the lasso regression and data cleaning notebooks. They are:

* `src/util.py`
* `src/cleaning\_util.py`
* `src/regression\_cv.py`


