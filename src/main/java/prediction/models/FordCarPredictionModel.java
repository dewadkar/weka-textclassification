package prediction.models;

import training.PredictionModel;

/**
 * Created by dewadkar on 11/08/16.
 */
public class FordCarPredictionModel extends PredictionModel {


    public static void main(String[] args) {
        PredictionModel predictionModel = new FordCarPredictionModel();
        String testData = null;
        System.out.printf(""+predictionModel.predict(testData));
    }
}
