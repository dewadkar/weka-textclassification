package training;

import preprocessing.DataSet;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.IOException;

/**
 * Created by dewadkar on 8/7/2016.
 */
public class PredictionModel {

    public static String DATA_SET_TRAIN_FILES = "resources/dataset/training/";
    public static String DATA_SET_TEST_FILES = "resources/dataset/testing";


    public static void main(String[] args) throws Exception {

        PredictionModel predictionModel = new PredictionModel();
        System.out.println("Wait ..  Converting directory into Data set");
        // convert the directory into a dataset
        DataSet securlyDataSet = new DataSet();
//        securlyDataSet.createTrainFileToDataSetDirForCar();
//        securlyDataSet.createTestFileToDataSetDirForCar();
//        securlyDataSet.createTrainFileToDataSetDirForCarRecommendation();
//        securlyDataSet.createTestFileToDataSetDirForCarRecommandation();

        System.out.println("Done ..  Created data set directory.");


        System.out.println("Wait ..  loading training dataset");
        Instances trainingDataSet = predictionModel.loadDirDataSetToWekaFormat(DATA_SET_TRAIN_FILES);
        System.out.println("Done ..  loaded dataset");

        System.out.println("Wait ..  converting data set to word vectors.");
        Filter stringToWordVectorFilter = predictionModel.filterStringToWordVectorType(trainingDataSet);
        Instances filteredTrainingDataSet = Filter.useFilter(trainingDataSet, stringToWordVectorFilter);
        System.out.println("Done ..  word vector file like arff file of weka created.");

        Classifier decisionTreeJ48 =predictionModel.getClassifier(filteredTrainingDataSet);

        System.out.println("Wait .. Evaluating Cross - Fold trained classifier.");
        predictionModel.evaluateCrossFoldModel(filteredTrainingDataSet, decisionTreeJ48);

        System.out.println("Wait .. Evaluating classifier for trained data.");
        predictionModel.evaluateTestingDataSetGivenTrainingDataSet(stringToWordVectorFilter, filteredTrainingDataSet, decisionTreeJ48);


    }

    private Classifier getClassifier(Instances filteredTrainingDataSet) throws Exception {
        System.out.println("Wait ..  processing attribute selection");
        ASEvaluation attributeFilter = new CfsSubsetEval();
        System.out.println("Done ..  Selected attributes and created training data set.");
        Classifier decisionTree = new J48();
        return buildClassifier(filteredTrainingDataSet, attributeFilter, decisionTree);
    }

    private GreedyStepwise isSearchBackWord() {
        GreedyStepwise attributeSearchModelForRanking = new GreedyStepwise();
        attributeSearchModelForRanking.setSearchBackwards(true);
        return attributeSearchModelForRanking;
    }

    private Classifier buildClassifier(Instances trainingData, ASEvaluation asEvaluation, Classifier classifier) throws Exception {
        System.out.println("Wait ..  Building classifier");

        AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
        attributeSelectedClassifier.setClassifier(classifier);
        attributeSelectedClassifier.setEvaluator(asEvaluation);
        attributeSelectedClassifier.setSearch(isSearchBackWord());

        classifier.buildClassifier(trainingData);
        System.out.println("\n\nClassifier model:\n\n" + classifier);
        System.out.println("Done ..  Trained classifier");
        return classifier;
    }

    private void evaluateTestingDataSetGivenTrainingDataSet(Filter filter, Instances dataFiltered, Classifier decisionTreeJ48) throws Exception {
        Instances testRawData = loadDirDataSetToWekaFormat(DATA_SET_TEST_FILES);
        Instances testingData = filteredData(filter, testRawData);
        evaluateTestingDataOnClassifier(dataFiltered, decisionTreeJ48, testingData);
    }

    private void evaluateCrossFoldModel(Instances dataFiltered, Classifier decisionTreeJ48) throws Exception {
        Evaluation evaluation = new Evaluation(dataFiltered);
        evaluation.crossValidateModel(decisionTreeJ48, dataFiltered, 4, new Debug.Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println("Done ..  Evaluation");
    }

    private StringToWordVector filterStringToWordVectorType(Instances dataRaw) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataRaw);
        filter.setStemmer(new SnowballStemmer());
        filter.setStopwords(new File("resources/stopword/engstopwords.txt"));
        return filter;
    }

    private Instances loadDirDataSetToWekaFormat(String dataSetDirPath) throws IOException {
        TextDirectoryLoader testLoader = new TextDirectoryLoader();
        testLoader.setDirectory(new File(dataSetDirPath));
        return testLoader.getDataSet();
    }

    private Instances filteredData(Filter filter, Instances testRawData) throws Exception {
        System.out.println("Wait .. Converting test data set to word vector.");

        Instances testingData = Filter.useFilter(testRawData, filter);

        System.out.println("Done .. Converted test data set to word ̄Avector.");
        return testingData;
    }

    private void evaluateTestingDataOnClassifier(Instances dataFiltered, Classifier decisionTreeJ48, Instances testingData) throws Exception {
        System.out.println("Wait .. Evaluating testing data on trained classifier.");

        Evaluation testEvaluation = new Evaluation(dataFiltered);
        testEvaluation.evaluateModel(decisionTreeJ48, testingData);
        System.out.println(testEvaluation.toSummaryString("\nResults\n======\n", false));

        System.out.println("Exit .. All processing done.");
    }

    public String predict(String testDataDirPath) {
        return null;
    }
}
