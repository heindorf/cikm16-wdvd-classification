Synopsis
========
The Wikidata Vandalism Detector 2016 (WDVD-2016) is a machine learning-based approach for automatic vandalism detection in Wikidata. It was developed as a joint project between Paderborn University and Bauhaus-Universität Weimar.

Paper
-----
This source code forms the basis for the Wikidata Vandalism Detector 2016 which was published at the CIKM 2016 conference. When using the code, please make sure to refer to it as follows:

Stefan Heindorf, Martin Potthast, Benno Stein, and Gregor Engels. Vandalism Detection in Wikidata. In Proceedings of the 25th ACM International Conference on Information and Knowledge Management (CIKM 16) (to appear), October 2016. ACM. <http://dx.doi.org/10.1145/2983323.2983740>

Classification Component
-------------------------
The classification component performs the classification and evaluation for the Wikidata Vandalism Detector 2016 (WDVD-2016). The feature extraction can be done with the corresponding [feature extraction component](https://github.com/heindorf/cikm16-wdvd-feature-extraction).

The code was tested with Python 3.5.1, 64 Bit under Windows 10.

### Installation
We recommend [Miniconda](http://conda.pydata.org/miniconda.html) for easy installation on many platforms.

1. Create new environment: `conda create --name wdvd16 python=3.5.1 --file wdvd-classification/requirements.txt`
2. Activate environment: `activate wdvd16`
3. Copy the [AUCCalculator](http://mark.goadrich.com/programs/AUC/) to the folder wdvd-classification/lib 


### Execution
Usage:

	python wdvd_classification.py FEATURES RESULTS

Given a FEATURES file (in bz2 format), splits the dataset, performs the classification and evaluation as described in the paper, and stores all results with the RESULTS prefix.

Example:

	python wdvd_classification.py wdvd16_features.csv.bz2 results/20160101_0000000/20160101_0000000

### Configuration
The constants in the file config.py control what parts of the code are executed, the caching behaviour as well as the level of parallelism.

Naturally, there is a tradeoff between maximum parallelism and minimum memory consumption. When executing all parts of the code with 16 parallel processes, about 128 GB RAM are required.

### Required Data
- [Feature file](http://groups.uni-paderborn.de/wdqa/cikm16/wdvd16-features.csv.bz2) as computed with the [feature extraction component](https://github.com/heindorf/cikm16-wdvd-feature-extraction)

### Known Issues
-  [WinError 1450] "Insufficient system resources exist to complete the requested service": This error sometimes occurs within the joblib library. Possible solutions:
   -   Defragmenting the hard disk might help: [https://support.microsoft.com/en-us/kb/967351](https://support.microsoft.com/en-us/kb/967351).
   -   Setting a registry key might help as well (not tested): [https://support.microsoft.com/en-us/kb/304101](https://support.microsoft.com/en-us/kb/304101)


Contact
=======
For questions and feedback please contact:

Stefan Heindorf, Paderborn University  
Martin Potthast, Bauhaus-Universität Weimar  
Benno Stein, Bauhaus-Universität Weimar  
Gregor Engels, Paderborn University

License
=======
Wikidata Vandalism Detector 2016 by Stefan Heindorf, Martin Potthast, Benno Stein, Gregor Engels is licensed under a MIT license.

