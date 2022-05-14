---
layout: page
title: Code
permalink: /Code/
---

All of the code for this project can be found [here](https://github.com/tulimid1/what-is-covering-me/tree/main). 

To recreate figures and analyses from paper, follow these steps:

1. Download the main branch of the [repo](https://github.com/tulimid1/what-is-covering-me/tree/main).

2. Create a new Python environment using [`Anaconda`](https://www.anaconda.com/) in the terminal.

```
cd .../what-is-covering-me/Code

conda env create -n CoverTypeProject -f CoverTypeProject.yml
```

3. Update the data path to your device. Go to `Code` folder of downloaded repo and edit line 8 of [ProcessData.py]() to .../what-is-covering-me/Data

4. Using the `CoverTypeProject` environment, run the following notebooks to recreate the following figures from the paper/presentation:

* Figure 2 : [____.ipynb]()
* Figure 3 : [Blackard_Dean_99.ipynb]()
* Figures 4-9 : [simulate2D_models.ipynb]()
* Figures 10, 12, 14, 16, 18, 19, 20 : [TheEnsemble.ipynb]()
* Figure 11 : [KNN_param_search.ipynb]()
* Figure 13 : [SVM_param_search.ipynb]()
* Figure 15 : [LDA_param_search.ipynb]()
* Figure 17 : [QDA_param_search.ipynb]()
