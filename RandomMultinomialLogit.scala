package RandomMultinomialLogit

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

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

  var PCCArray = ArrayBuffer[Double]()
  var wPCCArray = ArrayBuffer[Double]()
  var randomModels = ArrayBuffer[LogisticRegressionModel]()

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

  /**
    * Train a sequence of numSubsamples Logistic Regression-models.
    * @param input: Training data = RDD of LabeledPoint
    * @param numClasses: Number of classes
    * @param numFeatures: Number of features to randomly select for each model
    * @param numSubsamples: Number of subsamples of the input RDD to take = number of models in the forest
    */
  def runSequence(
                 input: RDD[LabeledPoint],
                 numClasses: Int,
                 numFeatures: Int,
                 numSubsamples: Int
                 ): Array[LogisticRegressionModel] = {

    // Specify model
    val RMLModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)

    // Initiate list to save models
    // val randomModels = new Array[LogisticRegressionModel](numSubsamples)

    // Determine total number of features once
    val allFeat = Array.range(0, input.first().features.size)

    // Relative prior frequencies of each class
    val relFreq = input.map(x => x.label)
      .countByValue().toArray
      .sortBy(x => x._1)
      .map(x => x._2.toDouble / input.count())
    // Weights to determine wPCC
    val weights = relFreq.map(x => (1 - x) / relFreq.map(x => 1 - x).sum)

    for (i <- List.range(0, numSubsamples)) {
      // Bootstrap sample
      val bootstrap = input.sample(withReplacement = withReplacement, fraction = subSamplingRate)

      // Select random set of features and save in featureList
      val subsetFeat = Random.shuffle(allFeat.toList).take(numFeatures).toArray

      bootstrap.foreach(observation => LabeledPoint(observation.label,
        Vectors.dense(subsetFeat.map(observation.features(_)))))

      // Train model
      val newRandomModel = RMLModel.run(bootstrap)
      randomModels += newRandomModel

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

    randomModels.toArray
  }

  /**
    * Aggregate predictions of an Array of LogisticRegressionModel.
    * Majority vote returns the class which was predicted most by all models,
    * adjusted majority vote returns the class with the highest mean
    * response probability over all models.
    *
    * @param testData: Vector of features
    * @param adjusted: Use adjusted majority vote (true) or regular majority vote (false)
    * @return Class index with majority vote
    */
  def aggregate(
                 testData: Vector,
                 adjusted: Boolean = false): Double = {

    if (!adjusted){
      // Return most occurring class id
      randomModels.map(x => x.predict(testData)).groupBy(identity).maxBy(_._2.length)._1
    } else {
      // Return class with highest mean probability
      randomModels
        .map(x => ClassificationUtility.predictPoint(testData, x)._2)
        .reduce((a, b) => a.zip(b).map(x => x._1 + x._2))
        .zipWithIndex
        .maxBy(_._1)._2
    }
  }

  /**
    * Take a subsample of n best models based on PCC performance.
    *
    * @param n: How many random models to select (default is 1, i.e. single best)
    */
  def takeBest(n: Int = 1): Array[LogisticRegressionModel] = {
    randomModels = PCC().zip(randomModels).sortBy(-_._1).map(_._2).take(n)
  }

}

