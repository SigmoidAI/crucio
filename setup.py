from distutils.core import setup
long_description = '''
Crucio is a python sci-kit learn inspired package for class imbalance. It use some classic methods for class balancing taking as parameters a data frame and the target column.\n
This version of crucio has the next methods of feature selection:\n
1) ADASYN\n
2) ICOTE (Immune Centroids Oversampling)\n
3) MTDF (Mega-Trend Difussion Function)\n
4) MWMOTE (Majority Weighted Minority Oversampling Technique)\n
5) SMOTE (Synthetic Minority Oversampling Technique)\n
6) SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)\n
7) SMOTETOMEK (Synthetic Minority Oversampling Technique + Tomek links for undersampling)\n
8) SMOTEENN (Synthetic Minority Oversampling Technique + ENN for undersampling)\n
9) SCUT (SMOTE and Clustered Undersampling Technique)\n
10) SLS (Safe-Level-Synthetic Minority Over-Sampling TEchnique)\n
11) TKRKNN (Top-K ReverseKNN)\n
All these methods takes the pandas Data Frame and y column to balance on.\n

How to use crucio\n
To use balancer from crucio you should just import the balancer from crucio in the following framework:\n
```from crucio import <class name>```\n
class names are written above in parantheses.\n
Next create a object of this algorithm (I will use ADASYN method as an example).\n
```method = ADASYN()```\n
To balance the dataset on the target column use the 'balance' function, using as parameters the pandas Data Frame and the column that you want to balance.\n
```new_dataframe = method.balance(df, 'target')```\n
Returned value is a new data frame with the target column balanced.\n

With love from Sigmoid.\n

We are open for feedback. Please send your impression to vpapaluta06@gmail.com\n

'''
setup(
  name = 'crucio',         # How you named your package folder (MyLib)
  packages = ['crucio'],   # Chose the same as "name"
  version = '0.1.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Crucio is a python sci-kit learn inspired package for class imbalance. It use some classic methods for class balancing taking as parameters a data frame and the target column.',   # Give a short description about your library
  long_description=long_description,
  author = 'YOUR NAME',                   # Type in your name
  author_email = 'vpapaluta06@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/ScienceKot/crucio.git',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ScienceKot/crucio/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'imbalanced learning', 'class balancing', 'python'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn',
          'statistics'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
  long_description_content_type='text/x-rst',
)