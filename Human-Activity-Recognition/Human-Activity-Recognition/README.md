# Project Title: Human Activity Recognition

## Project Description

[Link to dataset](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

Every year, an estimated 684,000 deaths occur globally due to [falls](https://www.who.int/news-room/fact-sheets/detail/falls). Hence, the motivation behind this project is to detect falls through postural changes.

Our approach to this problem is to focus on detecting 2 postural changes, **SIT_TO_LIE** and **STAND_TO_LIE**, as accurately as possible. We want to maximize accuracy and **minimize False Negatives** for both classes.

Machine Learning Techniques used:
1. Exploratory Data Analysis
2. Feature Engineering
3. Support Vector Machine
4. Logistic Regression
5. Decision Tree Classifier
6. Random Forest Classifiers
7. Artificial Neural Networks (ANN)
8. Long short-term Memory (LSTM)
9. Convolutional Neural Network (CNN)
10. Convolutional-LSTM Neural Netowrk (CNNLSTM)

## Evaluation of Models
| **Models** | **Accuracy** | **Recall** | **Precision** | **F1** |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SVM | 95.0%  | 88.0%  | 88.0%  | 88.0%  |
| Logistic Regression | 94.0%  | 83.0%  | 87.0%  | 85.0%  |
| RFC | 92.0%  | 82.0%  | 82.0%  | 83.0%  |
| ANN | 90.0% | 79.0%  | 82.0%  | 80.0% |
| CNN | 88.0% | 79.0%  | 81.0%  | 79.0% |
| RNN | 81.0% | 75.0%  | 78.0%  | 76.0% |
| CNNLSTM | 91.4% | 91.2% | 92.0% | 91.0% |

## Final CNNLSTM Model Architecture
![Screenshot 2022-12-05 225923](https://user-images.githubusercontent.com/101163864/205669061-4d0fd0ee-11fd-43a4-877f-908cff5c6020.png)

## Final Model
We **did not** choose the SVM or Logistic Regression model as the best classifiers because they were trained on features that have been engineered (using their statistics eg: mean, standard deviation, interquartile range etc.), which meant that the performance was **very specific to this particular experimental setting**. To generalize to a more practical use case, we relied on the raw signals. Hence, the CNNLSTM model was the best performing model on raw signals. However, although it was the best performing, it had a **higher False Negative rate**. Therefore, more work has to be done on this model for this specific problem setting. Nevertheless, it is still **effective and competent** in detecting human activities in general and can be used for other applications.

## Team Contributers
1. [Choo Wei Jie, Darren](https://github.com/dchoo99)
2. [Han Jiaxu](https://github.com/itsmejx)
3. [Daryl Ang](https://github.com/cambrian-dk)
4. [Teo Zhi Hao](https://github.com/Yttruire)
5. Sarah Tan
6. Kelvin Foo

## Mentor
Tian Fang

## Referenced Work:
1. https://www.cdc.gov/falls/data/fall-deaths.html
2. https://www.who.int/news-room/fact-sheets/detail/falls
3. https://www.who.int/news-room/fact-sheets/detail/falls
4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321307/
5. https://www.researchgate.net/publication/221313120
6. https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
7. https://www.kaggle.com/code/jillanisofttech/human-activity-recognition-with-smartphones-99-acc
8. https://www.kaggle.com/code/essammohamed4320/human-activity-recognition-scientific-prespective
9. https://www.kaggle.com/code/abeerelmorshedy/human-activity-recognition
10. https://www.kaggle.com/code/fahadmehfoooz/human-activity-recognition-with-neural-networks
11. A CNN-LSTM Approach to Human Activity Recognition by Ronald Mutegeki & Dong Seog Han
12. https://doi.org/10.1016/j.neucom.2015.07.085
13. StatQuest
14. https://stats.stackexchange.com/questions/369104/what-is-a-sensible-order-for-parameter-tuning-in-neural-networks
15. https://arxiv.org/abs/1609.04836
16. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT Press
17. https://link.springer.com/chapter/10.1007/978-3-642-10690-3_3
18. https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks
19. https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-network
20. https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks
21. https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
22. https://paperswithcode.com/method/convlstm
23. https://towardsdatascience.com/perturbation-theory-in-deep-neural-network-dnn-training-adb4c20cab1b
24. https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm
25. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4570352/
26. https://ieeexplore.ieee.org/document/6063364
27. https://keras.io/api
