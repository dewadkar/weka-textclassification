package training;

import preprocessing.DataSet;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

/**
 * Created by dewadkar on 8/7/2016.
 */
public class TrainTestClassifier {

    public static String DATA_SET_TRAIN_FILES = "resources/dataset/training/";
    public static String DATA_SET_TEST_FILES = "resources/dataset/testing";


    public static void main(String[] args) throws Exception {

        System.out.println("Wait ..  Converting directory into Data set");
        // convert the directory into a dataset
        DataSet securlyDataSet = new DataSet();
//        securlyDataSet.createTrainFileToDataSetDirForCar();
//        securlyDataSet.createTestFileToDataSetDirForCar();
//        securlyDataSet.createTrainFileToDataSetDirForCarRecommendation();
//        securlyDataSet.createTestFileToDataSetDirForCarRecommandation();

        System.out.println("Done ..  Created data set directory.");

        System.out.println("Wait ..  loading dataset");

        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File(DATA_SET_TRAIN_FILES));
        Instances dataRaw = loader.getDataSet();


        System.out.println("Done ..  loaded dataset");

//        System.exit(0);
        System.out.println("Wait ..  converting data set to word vectors.");

        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataRaw);
        Instances dataFiltered = Filter.useFilter(dataRaw, filter);

        System.out.println("Done ..  word vector file like arff file of weka created.");


        System.out.println("Wait ..  processing attribute selection");

        AttributeSelectedClassifier attributeSelectedClassifier = new AttributeSelectedClassifier();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);

        System.out.println("Done ..  Selected attributes and created training data set.");


        System.out.println("Wait ..  Building classifier");

        // train J48 and output model
        J48 decisionTreeJ48 = new J48();
        attributeSelectedClassifier.setClassifier(decisionTreeJ48);
        attributeSelectedClassifier.setEvaluator(eval);
        attributeSelectedClassifier.setSearch(search);


        decisionTreeJ48.buildClassifier(dataFiltered);
        System.out.println("\n\nClassifier model:\n\n" + decisionTreeJ48);
        System.out.println("Done ..  Trained classifier");


        System.out.println("Wait .. Evaluating trained classifier.");

        Evaluation evaluation = new Evaluation(dataFiltered);
        evaluation.crossValidateModel(decisionTreeJ48, dataFiltered, 4, new Debug.Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println("Done ..  Evaluation");



        System.out.println("Wait .. Loading test data set.");

        TextDirectoryLoader testLoader = new TextDirectoryLoader();
        testLoader.setDirectory(new File(DATA_SET_TRAIN_FILES));
        Instances testRawData = testLoader.getDataSet();

        System.out.println("Done .. Loaded test data set.");



        System.out.println("Wait .. Converting test data set to word vector.");

        StringToWordVector testFilter = new StringToWordVector();
        testFilter.setInputFormat(dataRaw);
        Instances testingData = Filter.useFilter(testRawData, testFilter);

        System.out.println("Done .. Converted test data set to word vector.");



        System.out.println("Wait .. Evaluating testing data on trained classifier.");

        Evaluation testEvaluation = new Evaluation(dataFiltered);
        testEvaluation.evaluateModel(decisionTreeJ48, testingData);
        System.out.println(testEvaluation.toSummaryString("\nResults\n======\n", false));

        System.out.println("Exit .. All processing done.");


    }
}
