package prediction.models;

import training.PredictionModel;

/**
 * Created by dewadkar on 11/08/16.
 */
public class FordCarLabelPredictionModel extends PredictionModel {
    public static void main(String[] args) {
        PredictionModel predictionModel = new FordCarLabelPredictionModel();
        String testData = null;
        System.out.printf(""+predictionModel.predict(testData));
    }
}
