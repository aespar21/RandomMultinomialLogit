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
