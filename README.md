## OLNLS: Online Learning for Noisy Labeled Streams

ACM Transactions on Knowledge Discovery from Data

### 🦸‍ Abstract
Online learning, characterized by its feature space's adaptability over time, has emerged as a flexible learning paradigm that has attracted widespread attention. However, existing online learning methods often overlook the distributional differences between instances and the presence of label noise in streaming data, thus significantly hindering the effectiveness and robustness of these algorithms. To overcome these challenges, we propose an online confidence learning algorithm for noisy labeled features, which aims to achieve robustness against arbitrary data streams and noisy labels.  It employs two new strategies: online confidence inference, which applies the principle of empirical risk minimization to identify inconsistencies in spatial distributions, and geometric structure learning, which utilizes dynamic instance confidence to compute disparities between instances and their labels. Empirical findings demonstrate that our label correction mechanism enhances classification accuracy more effectively across various types of noisy labels (i.e., symmetric, asymmetric, and flipped). Additionally, a case study on image datasets was conducted to illustrate in detail the effectiveness of our ONLNS algorithm.


![avatar](./Framework-OLNLS.png)


The comprehensive structure of this project is outlined as follows:

The dataset folder contains three different types of noise data streams, namely symmetric, asymmetric, and flipped noise stream. 

The source folder contains all the code.

The log is for saving last result(e.g. CAR).

The Result is CAR trends. (e.g. CAR, Figure)


## 📝 Getting Started
1.Clone the repository.
```
git clone 
```

2.Ensure that the environment meets the software's dependency requirements
```
pip install -r requirements.txt
```

3.Run the program on symmetric noisy labeled streaming
```
python main.py --noise_type sym --dataset wdbc
```

4.Run the program on asymmetric noisy labeled streaming
```
python main.py --noise_type asym --dataset wdbc
```

5.Run the program on flip noisy labeled streaming
```
python main.py --noise_type flip --dataset wdbc
```

### 📭 Maintainers
[Shengda Zhuo]
- ([zhuosd96@gmail.com](mailto:zhuosd96@gmail.com))
- 
