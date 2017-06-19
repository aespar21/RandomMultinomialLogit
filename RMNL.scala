package RandomMultinomialLogit

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Random Multinomial Logit
  * ------------------------
  * Constructs a bagged ensemble of
  * logistic regression models.
  *
  * Specify L1 or L2 regularization
  * through setRegParam and setElasticNetParam.
  *
  * runSequence is used to construct an ensemble of
  * logistic regression models, each based on a bootstrap
  * sample of the data and only considering a random
  * subset of the available variables.
  *
  * Assessing the accuracy on a validation set is done using
  * the aggregate-function,
  * returning an array of predicted classes.
  *
  * kFoldCrossValidate takes a dataset (RDD of LabeledPoint) and
  * returns an array of cross-validation accuracies of length k.
  */
class RMNL {

  var subSamplingRate: Double = 1.0
  var withReplacement: Boolean = true
  var oobEval: Boolean = false
  var numSubSamples: Int = 10

  var randomFeaturesArray = new Array[Array[Int]](numSubSamples)

  var PCCArray = new ArrayBuffer[Double]
  var wPCCArray = new ArrayBuffer[Double]

  var regParam = 0.0
  var elasticNetParam = 0.0

  val spark = SparkSession.builder().getOrCreate()
  import spark.implicits._

  /**
    * Set the regularization parameter. Default is 0.0.
    */
  def setRegParam(value: Double): this.type = {
    this.regParam = value
    this
  }

  /**
    * Set the ElasticNet mixing parameter.
    * For alpha = 0, the penalty is an L2 penalty.
    * For alpha = 1, it is an L1 penalty.
    * For 0 < alpha < 1, the penalty is a
    * combination of L1 and L2.
    * Default is 0.0 which is an L2 penalty.
    */
  def setElasticNetParam(value: Double): this.type = {
    this.elasticNetParam = value
    this
  }

  /**
    * Specify number of subsamples to take = number of
    * models in the ensemble
    */
  def setNumSubSamples(value: Int): this.type = {
    this.numSubSamples = value
    this.randomFeaturesArray = new Array[Array[Int]](numSubSamples)
    this
  }

  /**
    * Specify fraction of the training data used for model
    * building (default is 1.0)
    */
  def setSubSamplingRate(value: Double): this.type = {
    this.subSamplingRate = value
    this
  }

  /**
    * Specify whether to bootstrap with or without replacement
    * (default is true)
    */
  def setReplacement(value: Boolean): this.type = {
    this.withReplacement = value
    this
  }

  /**
    * Specify whether to calculate evaluation metrics from
    * out-of-bag data (default is false)
    */
  def setOobEval(value: Boolean): this.type = {
    this.oobEval = value
    this
  }

  /**
    * Get Percentage Correctly Classified
    * for all models trained
    */
  def PCC(): Array[Double] = {
    PCCArray.toArray
  }

  /**
    * Get weighted Percentage Correctly Classified
    * for all models trained
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
    * @param numFeatures: Number of features to
    *                   randomly select for each model
    */
  def runSequence(
                   input: RDD[LabeledPoint],
                   numClasses: Int,
                   numFeatures: Int
                 ): Array[LogisticRegressionModel] = {

    // Specify model with parameters
    val RMLModel = new LogisticRegression()
      .setRegParam(this.regParam)
      .setElasticNetParam(this.elasticNetParam)

    // Initiate list to save models
    val randomModels =
    new Array[LogisticRegressionModel](this.numSubSamples)

    // Determine total number of features once
    val allFeat = Array.range(0, input.first().features.size)

    // Relative prior frequencies of each class
    val relFreq = input.map(x => x.label)
      .countByValue().toArray
      .sortBy(x => x._1)
      .map(x => x._2.toDouble / input.count())
    // Weights to determine wPCC
    val weights = relFreq.map(x => 1 - x)

    for (i <- List.range(0, this.numSubSamples)) {
      // Bootstrap sample
      var bootstrap = input.sample(
        withReplacement = withReplacement,
        fraction = subSamplingRate)

      // Select random set of features and save in featureList
      val subsetFeat = Random.shuffle(allFeat.toList)
        .take(numFeatures).toArray
      randomFeaturesArray(i) = subsetFeat

      bootstrap = bootstrap.map(row => LabeledPoint(row.label,
        Vectors.dense(subsetFeat.map(row.features(_)))))

      // Train model
      val newRandomModel = RMLModel.fit(bootstrap.toDF())
      randomModels(i) = newRandomModel

      if (oobEval) {
        val newMetrics = oobEvaluation(input, bootstrap,
          subsetFeat, weights, newRandomModel)
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
                       subsetFeat: Array[Int],
                       weights: Array[Double],
                       model: LogisticRegressionModel):
    (Double, Double) = {

      val input_subsetFeat = input.map(row =>
        LabeledPoint(row.label,
          Vectors.dense(subsetFeat.map(row.features(_)))))

      // Out-of-bag data
      val oob = input_subsetFeat.subtract(bootstrap)

      val predictions = model.transform(oob.toDF())
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val PCC = evaluator.evaluate(predictions)

      val wPCC = predictions.select("label", "prediction").rdd
        .map(r => (r.get(0), r.get(1), r.get(0) == r.get(1)))
        .groupBy(r => r._1)
        .map(label => (label._1,
          label._2.map(c => if (c._3) 1.0 else 0.0).sum /
            label._2.toArray.length))
        .map(r => r._2 * weights(r._1.toString.dropRight(2).toInt))
        .sum() / numClasses

      (PCC, wPCC)
    }
    randomModels
  }

  /**
    * Aggregate predictions of an
    * Array of LogisticRegressionModel.
    * Majority vote returns the class
    * which was predicted most by all models,
    * adjusted majority vote returns
    * the class with the highest mean
    * response probability over all models.
    *
    * @param input: Array of LogisticRegressionModel
    *             (= ensemble of random models)
    * @param testData: Vector of features
    * @param adjusted: Use adjusted majority vote (true)
    *                or regular majority vote (false)
    * @return Class index with majority vote
    */
  def aggregate(
                 input: Array[LogisticRegressionModel],
                 testData: RDD[LabeledPoint],
                 adjusted: Boolean = false): Array[Double] = {
    // List to store predictions and probabilities for each model
    val predList = new Array[Array[Double]](input.length)
    var probList = (Array
      .fill[Array[Double]](testData.count().toInt)
      (Array.fill[Double](input.length)(0.0)))

    var predictions = new Array[Double](testData.count().toInt)
    var id = 0
    input.zip(this.randomFeaturesArray).foreach {
      case (model, randomFeatures) =>
      // Select correct subset of features
      val test = testData.map(row => LabeledPoint(row.label,
        Vectors.dense(randomFeatures.map(row.features(_)))))
      // Determine predictions + save in predList
      val predictions = model.transform(test.toDF())
      predList(id) = predictions
        .select("prediction")
        .map(r => r(0).asInstanceOf[Double])
        .collect()

      // Sum of class probabilities
      probList = predictions
        .select("probability")
        .collect()
        .map(_.get(0).toString)
        .map(_.drop(1).dropRight(1).split(",").map(_.toDouble))
        .zip(probList)
        .map(r => r._1.zip(r._2).map(k => k._2 + k._1))
      id += 1
    }
    if (!adjusted) {
      for (i <- 0 until testData.count().toInt) {
        predictions(i.toInt) = predList
          .map(x => x(i.toInt))
          .groupBy(identity)
          .maxBy(_._2.length)._1
      }
    } else {
      predictions = probList
        .map(_.zipWithIndex.maxBy(_._1)._2.toDouble)
    }
    predictions
  }

  /**
    * Cross-validation for the random multinomial logit.
    * @param input: RDD of LabeledPoint
    * @param numClasses: number of classes
    * @param numFeatures: number of random features
    *                   to select in each model
    * @param k: number of cross-validation folds
    * @param adjusted: whether to predict with adjusted
    *                or regular method
    * @param seed: seed to create folds
    * @return Array with accuracy for every model constructed
    */
  def kFoldCrossValidate(
                          input: RDD[LabeledPoint],
                          numClasses: Int,
                          numFeatures: Int,
                          k: Int = 10,
                          adjusted: Boolean = false,
                          seed: Int): Array[Double] = {
    val accuracyArray = ArrayBuffer[Double]()
    val folds = MLUtils.kFold(input, numFolds = k, seed = seed)

    folds.zipWithIndex.foreach {
      case ((training, validation), splitIndex) =>

        val trainingDataset = training.cache()
        val validationDataset = validation.cache()
        // Train RMNL
        val models = runSequence(
          trainingDataset, numClasses, numFeatures)
        trainingDataset.unpersist()
        // Determine accuracy on test set
        val predictions = aggregate(
          models, validationDataset, adjusted = adjusted)

        validationDataset.unpersist()
        accuracyArray += validationDataset.collect().map(_.label)
          .zip(predictions)
          .count(r => r._1 == r._2)
          .toDouble / validationDataset.count()
    }
    accuracyArray.toArray
  }
}
