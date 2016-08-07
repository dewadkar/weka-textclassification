package training;

import preprocessing.DataSet;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

/**
 * Created by dewadkar on 8/7/2016.
 */
public class TrainTestClassifier {



    public static void main(String[] args) throws Exception {

        // convert the directory into a dataset
        DataSet securlyDataSet = new DataSet();
//        securlyDataSet.createTrainFileToDataSetDirForCar();
//        securlyDataSet.createTestFileToDataSetDirForCar();
//        securlyDataSet.createTrainFileToDataSetDirForCarRecommendation();
//        securlyDataSet.createTestFileToDataSetDirForCarRecommandation();


        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File(securlyDataSet.DATA_SET_TRAIN_FILES));
        Instances dataRaw = loader.getDataSet();
        loader.getOutputFilename();
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataRaw);
        Instances dataFiltered = Filter.useFilter(dataRaw, filter);


//        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
//        CfsSubsetEval eval = new CfsSubsetEval();
//        GreedyStepwise search = new GreedyStepwise();
//        search.setSearchBackwards(true);

        AttributeSelection attributeSelection = new AttributeSelection();  // package weka.filters.supervised.attribute!
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(search);
        attributeSelection.setInputFormat(dataFiltered);
        // generate new data
        Instances newData = Filter.useFilter(dataFiltered,attributeSelection);
        // train J48 and output model
        J48 baseClassifier = new J48();
        baseClassifier.buildClassifier(newData);
        System.out.println("\n\nClassifier model:\n\n" + baseClassifier);
    }
}
