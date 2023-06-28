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

/**
 * Single map-reduce job
 * Mapper: sends subset of data to the reducer.
 * Each reduce tasks trains one model and then saves it.
 */
public class EnsembleTraining extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(EnsembleTraining.class);
    private static final String INTERMEDIATE_PATH = "intermediate_model_output";

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {


        /**
         * The purpose of the mapper is to send each input record
         * to a reduce task. Each map call decides where to send the input
         * record based on randomisaztion. Since we want to train a 100 models,
         * we generate a random number between 0 and 100 and send the input record
         * to that reduce task. So each reduce task receives a random sample of the data.
         * Reason for doing this randomization is to ensure the the sample is representative of the
         * population. And this also prevents the need of having to determine where the record should go
         * using multiple if statements for ranges eg: if key between x and y, record to reduce task Z.
         * So in the emit, the "key" is the random number generated and the "value" is the entire record.
         *
         * @param key is the offset of the record
         * @param value is the input record
         * @param context
         * @throws java.io.IOException
         * @throws InterruptedException
         */
        @Override
        public void map(final Object key, final Text value, final Context context) throws java.io.IOException, InterruptedException {

            //Defining the start and the end range for generating the random number
            //range depends on the number of models you want to train.
            //if you want to train 100 models, range should be 0-100.
            int start_range = 0;
            int end_range = 100;

            // Generating a random number based on the start and the end range
            Random random = new Random();
            // Assigning the random number to the key variable
            int key1 = random.nextInt(end_range - start_range + 1) + start_range;

            // For each datapoint, we assign the random number generated above as the key so that the input record goes to that reducer
            context.write(new Text(String.valueOf(key1)), value);
        }

    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);

        /**
         * Key eg: 2, values: all the input record sent to the reduce task with this key.
         * Over here, we receive all the input records and we will now train a decision tree model.
         * Each reduce task trains one decision tree model (Weka j48) using the subset of data received
         * and then saves the model to HDFS once completed.
         * @param key
         * @param values
         * @param context
         * @throws java.io.IOException
         * @throws InterruptedException
         */
        @Override
        public void reduce(final Text key, final Iterable<Text> values, final Context context) throws java.io.IOException, InterruptedException {

            // Defining a List of List to store all the records.
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

                //save the model
                String file_name = new String(INTERMEDIATE_PATH + "/dt_" + key.toString() + ".model");

                SerializationHelper.write(file_name, classifier);
                
            } catch (Exception e) {
                logger.error("Failed to train the model.", e);
                throw new IOException("Failed to train the model.", e);
            }

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
