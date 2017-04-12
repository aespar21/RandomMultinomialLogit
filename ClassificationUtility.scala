package RandomMultinomialLogit

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vector

object ClassificationUtility {

  def predictPoint(dataMatrix: Vector, model: LogisticRegressionModel):
  (Double, Array[Double]) = {

    require(dataMatrix.size == model.numFeatures)
    val dataWithBiasSize: Int = model.weights.size / (model.numClasses - 1)
    val weightsArray: Array[Double] = model.weights match {
      case dv: DenseVector => dv.values
      case _ =>
        throw new IllegalArgumentException(
          s"weights only supports dense vector but got type ${model.weights.getClass}.")
    }
    var bestClass = 0
    var maxMargin = 0.0
    val withBias = dataMatrix.size + 1 == dataWithBiasSize
    val classProbabilities: Array[Double] = new Array[Double](model.numClasses)
    (0 until model.numClasses - 1).foreach { i =>
      var margin = 0.0
      dataMatrix.foreachActive { (index, value) =>
        if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
      }
      // Intercept is required to be added into margin.
      if (withBias) {
        margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
      }
      if (margin > maxMargin) {
        maxMargin = margin
        bestClass = i + 1
      }

      classProbabilities(i+1) = 1.0 / (1.0 + Math.exp(-(margin - maxMargin)))
    }

    (bestClass.toDouble, classProbabilities)
  }
}
