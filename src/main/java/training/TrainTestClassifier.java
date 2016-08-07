package training;

import preprocessing.DataSet;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

/**
 * Created by dewadkar on 8/7/2016.
 */
public class TrainTestClassifier {



    public static void main(String[] args) throws Exception {

        // convert the directory into a dataset
        DataSet securlyDataSet = new DataSet();
        securlyDataSet.createTrainFileToDataSetDirForCar();
        securlyDataSet.createTestFileToDataSetDirForCar();
        securlyDataSet.createTrainFileToDataSetDirForCarRecommendation();
        securlyDataSet.createTestFileToDataSetDirForCarRecommandation();


        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File(securlyDataSet.DATA_SET_TRAIN_FILES));
        Instances dataRaw = loader.getDataSet();
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataRaw);
        Instances dataFiltered = Filter.useFilter(dataRaw, filter);

        // train J48 and output model
        J48 classifier = new J48();
        classifier.buildClassifier(dataFiltered);
        System.out.println("\n\nClassifier model:\n\n" + classifier);
    }
}
