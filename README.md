# crucio
Crucio is a python sci-kit learn inspired package for class imbalance. It use some classic methods for class balancing taking as parameters a data frame and the target column.

This version of kydavra has the next methods of feature selection:
* ADASYN.
* ICOTE (Immune Centroids Oversampling).
* MTDF (Mega-Trend Difussion Function).
* MWMOTE (Majority Weighted Minority Oversampling Technique).
* SMOTE (Synthetic Minority Oversampling Technique).
* SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous).
* SMOTETOMEK (Synthetic Minority Oversampling Technique + Tomek links for undersampling).
* SMOTEENN (Synthetic Minority Oversampling Technique + ENN for undersampling).
* SCUT (SMOTE and Clustered Undersampling Technique).
* SLS (Safe-Level-Synthetic Minority Over-Sampling TEchnique).
* TKRKNN (Top-K ReverseKNN).\

All these methods takes the pandas Data Frame and y column to balance on.

How to use crucio

To use balancer from crucio you should just import the balancer from crucio in the following framework:\
```from crucio import <class name>```\
class names are written above.\
Next create a object of this algorithm (I will use ADASYN method as an example).
```method = ADASYN()```

To balance the dataset on the target column use the ‘balance’ function, using as parameters the pandas Data Frame and the column that you want to balance.

```new_dataframe = method.balance(df, 'target')```

Returned value is a new data frame with the target column balanced.

With love from Sigmoid.

We are open for feedback. Please send your impression to papaluta.vasile@isa.utm.md
