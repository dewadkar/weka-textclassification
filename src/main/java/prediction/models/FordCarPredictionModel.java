package prediction.models;

import org.apache.commons.io.FileUtils;
import training.PredictionModel;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by dewadkar on 11/08/16.
 */
public class FordCarPredictionModel extends PredictionModel {


    public static void main(String[] args) {
        PredictionModel predictionModel = new FordCarPredictionModel();
        try {
            Instances trainingData = predictionModel.productTrainedData();
            int numAttributes= 45;
            Instances attributeSelectedByChiSquare = predictionModel.chisquareAttributeSelection(trainingData, numAttributes);
            Instances attributeSelectedByInfoGain = predictionModel.infoGainAttributeSelection(trainingData, numAttributes);
            Instances attributeSelectedByGainRatio = predictionModel.gainRatioAttributeSelection(trainingData, numAttributes);
            File file = new File("resources/product/selectedAttributesLabelData.txt");
            for (int i = 0; i < numAttributes; i++) {
                String data = attributeSelectedByChiSquare.attribute(i).name() + " "+attributeSelectedByInfoGain.attribute(i).name()+ " "+attributeSelectedByGainRatio.attribute(i).name()+"\n";
                System.out.print(data);
                FileUtils.writeStringToFile(file,data);
            }

            List<Classifier> classifiers = new LinkedList<Classifier>();
            classifiers.add(new NaiveBayes());
            classifiers.add(new J48());
            classifiers.add(new LibSVM());
            List<Instances> instances = new LinkedList<Instances>();
            instances.add(attributeSelectedByChiSquare);
            instances.add(attributeSelectedByGainRatio);
            instances.add(attributeSelectedByInfoGain);
            File evaluationFile = new File("resources/evaluation/product/classifierEvaluation.txt");
            for (Classifier classifier: classifiers) {
                for (Instances instance : instances) {
                    Classifier classifierModel =  predictionModel.buildClassifier(instance, classifier);
                    Debug.saveToFile("resources/models/product/"+classifier.toString(), classifier);
                    Evaluation evaluation = predictionModel.testingClassifierForProductData(classifierModel);
                    FileUtils.writeStringToFile(evaluationFile,evaluation.toSummaryString());
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
