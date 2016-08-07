package preprocessing;

import org.apache.commons.io.FileUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import sun.nio.cs.StandardCharsets;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by dewadkar on 8/5/2016.
 */
public class DataSet {
    static String DATA_SET_TEST = "C:\\Users\\IBM_ADMIN\\IdeaProjects\\nlp\\resources\\hw2\\auto.dat.test";
    static String DATA_SET_TRAIN = "C:\\Users\\IBM_ADMIN\\IdeaProjects\\nlp\\resources\\hw2\\auto.dat.test";

    static final Pattern TAG_REGEX = Pattern.compile("(<DOC>)(.+?)(</DOC>)");

    public File createCSV(String raw_data_file_path) {

        documents();
        return  null;
    }

    private void documents() {
        try {
            String content = FileUtils.readFileToString(new File(DATA_SET_TEST));
            List<String> tagValues = new ArrayList<String>();
            int i =0;
            int temp=0;
            while((i=(content.indexOf("<DOC>",i)+1))>0){
                if(temp==0){
                    temp = i;
                }else{
                    System.out.println(temp+" -- "+i);
                tagValues.add(content.substring(temp,i));
                }
            }
            tagValues.add(content.substring(temp,content.length()));
            System.out.println(" Total Documents " + tagValues.size());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String fileContent(String fileName) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();
            while (line != null) {
                sb.append(line);
            }
            return sb.toString();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) {
        DataSet dataSet = new DataSet();
        dataSet.createCSV(DATA_SET_TRAIN);
    }
}
