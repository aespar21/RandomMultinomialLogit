# RandomMultinomialLogit

This Spark/Scala-package contains an implementation of the random multinomial logit. This method permits the use of multinomial logistic regression when dealing with huge feature spaces (in particular when p >> n). An overview of the technique and how to use the package is given below.

## Random Multinomial Logistic Regression
While the robustness of multinomial regression is widely appreciated, problems can occur when faced with huge feature spaces. The random multinomial logit tries to overcome this issue by constructing many logistic regression models instead of one, and does so by borrowing its structure from the random forest framework:
* An ensemble of logistic regression models is constructed, each based on a bootstrapped sample (*random* sample with replacement) of the original data
* Each model is trained using only a *random* subset of the total set of features 

To make predictions for new instances, the results of all models in the ensemble can be summarized in two ways:
* The *majority vote* principle, where each model in the ensemble can cast a vote (which class to assign to the observation)
* *Adjusted majority vote* is an extension of the latter by using the response probabilities for each class of each model in the ensemble. The class with the highest mean repsonse probability over all models in the ensemble will be assigned to the new observation

## RandomMultinomialLogit: Minimal Example
A random multinomial logit-model is initiated as follows. Also specify the number of individual models to be constructed in the model in this statement, in this case 10. 
```scala
val model = new RMNL()
  .setNumSubSamples(10)
```
The model is then fitted to a set of training data (available as an RDD of LabeledPoint). Also specify the total number of classes and the number of random features to select in each individual model. `fit` is an Array of LogisticRegressionModel.
```scala
val fit = model.runSequence(
  input = train,
  numClasses = 3,
  numFeatures = 50)
```
Aggregating the predictions of all constituent models is possible as follows. Test data should be contained in an RDD of LabeledPoint. The adjusted-Boolean indicated whether to aggregate using majority vote or adjusted majority vote (as explained above). `predictions` is an Array of predicted classes for each test observation.
```scala
val predictions = model.aggregate(
  input = fit,
  testData = test,
  adjusted = true)
```
Cross-validation is also available. Specify the number of folds to use, the total number of classes, the number of random features to select in each individual model, and whether to aggregate using majority vote or adjusted majority vote. Also provide a seed to use for splitting the data in `k` folds. `CV-accuracy` is an Array of accuracy for each fold (based on a model constructed of the other folds).
```scala
val CV_accuracy = model.kFoldCrossValidate(
  input = data,
  numClasses = 3,
  k = 10,
  adjusted = false,
  seed = 12345)
```
