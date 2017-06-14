package RandomMultinomialLogit

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.util.MLUtils

/**
  * Random Multinomial Logit Algorithm:
  * Multi-class predictions based on a random forest
  * of multinomial logit-models constructed from a
  * bootstrapped sample and a random subset of features.
  */

object RandomMultinomialLogit extends LogisticRegressionWithLBFGS {

  var subSamplingRate: Double = 1.0
  var withReplacement: Boolean = true
  var oobEval: Boolean = false
  var numSubSamples: Int = 10

  var randomFeaturesArray = new Array[Array[Int]](numSubSamples)

  var PCCArray = ArrayBuffer[Double]()
  var wPCCArray = ArrayBuffer[Double]()

  /**
    * Specify number of subsamples to take = number of models in the ensemble
    */
  def setNumSubSamples(value: Int): this.type = {
    this.numSubSamples = value
    this.randomFeaturesArray = new Array[Array[Int]](numSubSamples)
    this
  }

  /**
    * Specify fraction of the training data used for model building (default is 1.0)
    */
  def setSubSamplingRate(value: Double): this.type = {
    this.subSamplingRate = value
    this
  }

  /**
    * Specify whether to bootstrap with or without replacement (default is true)
    */
  def setReplacement(value: Boolean): this.type = {
    this.withReplacement = value
    this
  }

  /**
    * Specify whether to calculate evaluation metrics from out-of-bag data (default is false)
    */
  def setOobEval(value: Boolean): this.type = {
    this.oobEval = value
    this
  }

  /**
    * Get Percentage Correctly Classified for all models trained
    */
  def PCC(): Array[Double] = {
    PCCArray.toArray
  }

  /**
    * Get weighted Percentage Correctly Classified for all models trained
    */
  def wPCC(): Array[Double] = {
    wPCCArray.toArray
  }

  def randomFeatures(): Array[Array[Int]] = {
    randomFeaturesArray
  }

  /**
    * Train a sequence of numSubsamples Logistic Regression-models.
    * @param input: Training data = RDD of LabeledPoint
    * @param numClasses: Number of classes
    * @param numFeatures: Number of features to randomly select for each model
    */
  def runSequence(
                 input: RDD[LabeledPoint],
                 numClasses: Int,
                 numFeatures: Int
                 ): Array[LogisticRegressionModel] = {

    // Specify model
    val RMLModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)

    // Initiate list to save models
    val randomModels = new Array[LogisticRegressionModel](this.numSubSamples)

    // Determine total number of features once
    val allFeat = Array.range(0, input.first().features.size)

    // Relative prior frequencies of each class
    val relFreq = input.map(x => x.label)
      .countByValue().toArray
      .sortBy(x => x._1)
      .map(x => x._2.toDouble / input.count())
    // Weights to determine wPCC
    val weights = relFreq.map(x => (1 - x) / relFreq.map(x => 1 - x).sum)

    for (i <- List.range(0, this.numSubSamples)) {
      // Bootstrap sample
      var bootstrap = input.sample(withReplacement = withReplacement, fraction = subSamplingRate)

      // Select random set of features and save in featureList
      val subsetFeat = Random.shuffle(allFeat.toList).take(numFeatures).toArray
      randomFeaturesArray(i) = subsetFeat

      bootstrap = bootstrap.map(row => LabeledPoint(row.label,
        Vectors.dense(subsetFeat.map(row.features(_)))))

      // Train model
      val newRandomModel = RMLModel.run(bootstrap)
      randomModels(i) = newRandomModel

      if (oobEval) {
        val newMetrics = oobEvaluation(input, bootstrap, weights, newRandomModel)
        PCCArray += newMetrics._1
        wPCCArray += newMetrics._2
      }
      }

    /**
      * Determines out-of-bag metrics:
      * PCC = Percentage Correctly Classified
      * wPCC = weighted PCC
      * @param input: All training data
      * @param bootstrap: Bootstrap sample from training data
      * @param weights: Array with weight for each class
      * @param model: Logistic regression model to evaluate
      * @return PCC and wPCC
      */
    def oobEvaluation(
                       input: RDD[LabeledPoint],
                       bootstrap: RDD[LabeledPoint],
                       weights: Array[Double],
                       model: LogisticRegressionModel): (Double, Double) = {
      // Out-of-bag data
      val oob = input.subtract(bootstrap)

      // Determine absolute class frequencies in out-of-bag sample
      val absOOBFreq = oob
        .map(x => x.label)
        .countByValue().toArray
        .sortBy(x => x._1)
        .map(x => x._2.toInt)

      val labelAndPreds = oob.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }

      val PCC = labelAndPreds.filter(r => r._1 == r._2).count.toDouble / oob.count()

      // Needs testing!
      //val wPCC = labelAndPreds.filter(r => r._1 == r._2).countByKey()
      //  .toArray.sortBy(x => x._1).map(x => x._2.toDouble / absOOBFreq(x._1.toInt) * weights(x._1.toInt)).sum
      val wPCC = 0.0
      (PCC, wPCC)
    }

    randomModels
  }

  /**
    * Aggregate predictions of an Array of LogisticRegressionModel.
    * Majority vote returns the class which was predicted most by all models,
    * adjusted majority vote returns the class with the highest mean
    * response probability over all models.
    *
    * @param input: Array of LogisticRegressionModel (= ensemble of random models)
    * @param testData: Vector of features
    * @param adjusted: Use adjusted majority vote (true) or regular majority vote (false)
    * @return Class index with majority vote
    */
  def aggregate(
                 input: Array[LogisticRegressionModel],
                 testData: Vector,
                 adjusted: Boolean = false): Double = {

    if (!adjusted){
      // Return most occurring class id
      //input.map(x => x.predict(testData)).groupBy(identity).maxBy(_._2.length)._1
      input.zip(this.randomFeaturesArray).map(x => x._1.predict(Vectors.dense(x._2.map(testData(_)))))
        .groupBy(identity).maxBy(_._2.length)._1
    } else {
      // Return class with highest mean probability
      input
        .map(x => ClassificationUtility.predictPoint(testData, x)._2)
        .reduce((a, b) => a.zip(b).map(x => x._1 + x._2))
        .zipWithIndex
        .maxBy(_._1)._2
    }
  }

  /**
    * Take a subsample of n best models based on PCC performance.
    *
    * @param input: Array of LogisticRegressionModel (= ensemble of random models)
    * @param n: How many random models to select (default is 1, i.e. single best)
    */
  def takeBest(
                input: Array[LogisticRegressionModel],
                n: Int = 1): Array[LogisticRegressionModel] = {
    PCC().zip(input).sortBy(-_._1).map(_._2).take(n)
  }

  def kFoldCrossValidate(
                     input: RDD[LabeledPoint],
                     numClasses: Int,
                     numFeatures: Int,
                     k: Int = 10,
                     seed: Int): Array[Double] = {
    val accuracyArray = ArrayBuffer[Double]()
    val folds = MLUtils.kFold(input, numFolds = k, seed = seed)

    folds.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = training.cache()
      val validationDataset = validation.cache()
      // Train RMNL
      val models = runSequence(trainingDataset, numClasses, numFeatures)
      trainingDataset.unpersist()
      // Determine accuracy on test set
      val labelsAndPreds = validationDataset.map { point =>
        val prediction = RandomMultinomialLogit.aggregate(models, point.features)
        (point.label, prediction)}
      validationDataset.unpersist()

      accuracyArray += labelsAndPreds.filter(r => r._1 == r._2).count.toDouble / validationDataset.count()
    }
    accuracyArray.toArray
  }

}

