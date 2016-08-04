package testing;

import org.junit.Ignore;
import org.junit.Test;
import training.AutoClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.*;

import static org.junit.Assert.assertNotNull;

/**
 * Created by dewadkar on 8/4/2016.
 */
public class ClassifierTest
{
    private static final String DATA_SET ="resources/weather.txt" ;

    @Test @Ignore
    public void testClassfierReadDataFileFunction(){
//        assertNotNull(new AutoClassifier().readDataFile("resources/weather.txt"));
    }

    @Test
    public void testClassifierJ48ClassifyFunction(){
        Classifier[] modelNameJ48 = {new J48()};
        AutoClassifier autoClassifier=new AutoClassifier();
        String dataFilePath="resources/weather.txt";
        Instances[][] instanceData = null;
        try {
            instanceData = autoClassifier.dataInstances(dataFilePath);
            autoClassifier.validateClassifiersForTrainingData(instanceData,modelNameJ48);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
