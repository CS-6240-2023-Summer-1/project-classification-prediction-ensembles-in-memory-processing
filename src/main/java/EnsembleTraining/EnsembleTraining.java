package EnsembleTraining;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.core.SerializationHelper;
import weka.core.Utils;
import java.io.FileWriter;
import java.io.IOException;

public class EnsembleTraining extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(EnsembleTraining.class);
    private static final String INTERMEDIATE_PATH = "intermediate_model_output";

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void map(final Object key, final Text value, final Context context) throws java.io.IOException, InterruptedException {
            //logger.info("In MAPPER");

            // Defining the start and the end range for generating the random number
            int start_range = 0;
            int end_range = 100;

            // Generating a random number based on the start and the end range
            Random random = new Random();

            // Assigning the random number to the key variable
            int key1 = random.nextInt(end_range - start_range + 1) + start_range;

            // For each datapoint, we assign the random number generated above as the key so that the input data goes to the same reducer
            // Emitting the key and the value in which values is the entire row that includes features and label
            context.write(new Text(String.valueOf(key1)), value);
        }

    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);


        @Override
        public void reduce(final Text key, final Iterable<Text> values, final Context context) throws java.io.IOException, InterruptedException {

            // Defining a List of List to store the records in which each nested row represents a row
            List<List<String>> allRecords = new ArrayList<>();

            // Iterating over the values emitted from the mapper (rows)
            for (final Text val : values) {
                // Splitting the row with ','
                String[] record = String.valueOf(val).split(",");

                // Storing the "," separated string as a list
                List<String> recordList = new ArrayList<>(Arrays.asList(record));

                // Adding this list into the allrecords
                allRecords.add(recordList);
            }

            // Defining the attributes list
            List<Attribute> attributes = new ArrayList<>();

            // Iterating over each column and adding features to the attributes list
            for (int i = 0; i < allRecords.get(0).size() - 1; i++) {
                attributes.add(new Attribute("feature" + (i + 1)));
            }

            // Defining the classLabels
            List<String> classLabels = new ArrayList<>();

            // Specifying the classes into the classLabels
            classLabels.add("0");
            classLabels.add("1");

            // Adding the classLabels into the attributes list
            Attribute classAttribute = new Attribute("class", classLabels);
            attributes.add(classAttribute);

            // Defining the dataset instance
            Instances dataset = new Instances("Dataset", (ArrayList<Attribute>) attributes, allRecords.size());

            // Setting the label index
            dataset.setClassIndex(attributes.size() - 1);

            // Iterating over each records
            for (List<String> record : allRecords) {
                // Defining new instance for each row of the dataset
                weka.core.Instance instance = new DenseInstance(attributes.size());

                // Iterating over all the features in the row
                for (int i = 0; i < record.size(); i++) {
                    // For each feature in the row, setting the feature value into the instance
                    instance.setValue(i, Double.parseDouble(record.get(i)));
                }

                // Adding the generated instance (row) into the dataset
                dataset.add(instance);
            }

            // Setting the label index
            dataset.setClassIndex(attributes.size() - 1);

            try {
                // Building the model
                J48 classifier = new J48();

                // Training the model with the given dataset
                classifier.buildClassifier(dataset);

                //System.out.println("----------------------");
                //System.out.println(key);

                // Printing the tree
                //System.out.println(classifier);

                String file_name = new String(INTERMEDIATE_PATH + "/dt_" + key.toString() + ".model");

                SerializationHelper.write(file_name, classifier);
                //saveModelParametersToCSV(classifier, "/Users/ashirm1999/Desktop/Large_Scale/project-classification-prediction-ensembles-in-memory-processing/input/model_parameters.csv");

                
            } catch (Exception e) {
                logger.error("Failed to train the model.", e);
                throw new IOException("Failed to train the model.", e);
            }

        }

        private void saveModelParametersToCSV(J48 classifier, String filePath) throws IOException {
            FileWriter writer = new FileWriter(filePath);
            String[] options = classifier.getOptions();
            writer.append("Option,Value\n");
            for (String option : options) {
                String[] parts = option.split(" ");
                if (parts.length > 1) {
                    String optionName = parts[0];
                    String optionValue = parts[1];
                    writer.append(optionName).append(",").append(optionValue).append("\n");
                }
            }
            writer.flush();
            writer.close();
        }
    }

    @Override
    public int run(final String[] args) throws Exception {
        final Configuration conf = getConf();
        final Job job = Job.getInstance(conf, "Word Count");
        job.setJarByClass(EnsembleTraining.class);

        final Configuration jobConf = job.getConfiguration();
        jobConf.set("mapreduce.output.textoutputformat.separator", "\t");

        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(INTERMEDIATE_PATH));
        //FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(final String[] args) {
        if (args.length != 2) {
            throw new Error("Two arguments required:\n<input-dir> <output-dir>");
        }

        try {
            ToolRunner.run(new EnsembleTraining(), args);
        } catch (final Exception e) {
            logger.error("", e);
        }
    }
}
