package Dataset_Split;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class DatasetEntry {
    private String data;
    private String label;

    public DatasetEntry(String data, String label) {
        this.data = data;
        this.label = label;
    }

    public String getData() {
        return data;
    }

    public String getLabel() {
        return label;
    }
}

/**
 * The purpose of this class is to do a train-test split. Given a dataset
 * we do a 80-20 split here. You can change this bu specifying the parameters.
 * The prpgram takes the inputs, does the splits and then saves the respective files.
 */

public class TrainTestSplit {
    public static void main(String[] args) {
        String csvFile = "/Users/ashirm1999/Desktop/Large_Scale/project-classification-prediction-ensembles-in-memory-processing/Final_Higgs_Data.csv"; // path of CSV file
        String delimiter = ","; // delimiter in CSV file
        String trainFolder = "input/train"; // Folder to save train CSV
        String testFolder = "input/test"; // Folder to save test CSV

        List<DatasetEntry> dataset = readDatasetFromCSV(csvFile, delimiter);

        // Shuffling the dataset for randomness
        Collections.shuffle(dataset);

        // Setting the ratio for train and test
        double trainRatio = 0.8;

        // calculating the train size using the ratio
        int trainSize = (int) (trainRatio * dataset.size());

        List<DatasetEntry> trainSet = dataset.subList(0, trainSize);
        List<DatasetEntry> testSet = dataset.subList(trainSize, dataset.size());

        // Creating the train and test folders
        createFolder(trainFolder);
        createFolder(testFolder);

        // Saving the train and test csv into specific folders
        saveDatasetToCSV(trainSet, trainFolder + "/train_dataset.csv", delimiter);
        saveDatasetToCSV(testSet, testFolder + "/test_dataset.csv", delimiter);

        // Printing the train and test size
        System.out.println("Train Data Size: " + trainSet.size());
        System.out.println("Test Data Size: " + testSet.size());
    }

    // Reading the data from the CSV File
    private static List<DatasetEntry> readDatasetFromCSV(String csvFile, String delimiter) {
        List<DatasetEntry> dataset = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;

            // Reading until the line is not null
            while ((line = br.readLine()) != null) {

                // Splitting the line by ,
                String[] parts = line.split(delimiter);
                if (parts.length >= 2) {
                    // Initializing the feature list to store all the 28 features
                    List<String> features = new ArrayList<>();

                    // Iterating over all the parts to extract the feature and append into the list
                    for (int i = 1; i <= 10; i++) {
                        features.add(parts[i]);
                    }

                    // Extracting the label which is the first column in the given dataset
                    String label = parts[0];

                    // Writing the data
                    dataset.add(new DatasetEntry(features.toString(), label));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataset;
    }

    // Saving the dataset back into the specific directory
    private static void saveDatasetToCSV(List<DatasetEntry> dataset, String csvFile, String delimiter) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvFile))) {
            for (DatasetEntry entry : dataset) {

                String line = entry.getData().replaceAll("\\[|\\]", "") + delimiter + entry.getLabel();
                bw.write(line);
                bw.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Creating the train and test folder if they do not exist
    private static void createFolder(String folderPath) {
        File folder = new File(folderPath);
        if (!folder.exists()) {
            boolean created = folder.mkdirs();
            if (!created) {
                System.err.println("Failed to create folder: " + folderPath);
            }
        }
    }
}
