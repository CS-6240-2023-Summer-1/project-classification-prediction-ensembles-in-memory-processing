package EnsembleTraining;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;
import java.io.File;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.*;

public class ModelTesting extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(ModelTesting.class);
    private static List<Classifier> models = new ArrayList<>();
    enum Accuracy {
        TRUE_POSITIVE_COUNT,
        TRUE_NEGATIVE_COUNT,
        FALSE_POSITIVE_COUNT,
        FALSE_NEGATIVE_COUNT
    }
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            //Configuration conf = new Configuration();

            //Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);

            URI[] cacheFiles = context.getCacheFiles();
            String line;
            FileSystem fs;

            try {
                fs = FileSystem.get(new URI(context.getConfiguration().get("cache.fs")), context.getConfiguration());
                Path directoryPath = new Path(cacheFiles[0].toString());

                // List all the files inside the directory
                FileStatus[] fileStatuses = fs.listStatus(directoryPath);

                // Iterate over each file
                for (FileStatus fileStatus : fileStatuses) {
                    Path filePath = fileStatus.getPath();

                    // Check if the file name ends with ".model"
                    if (filePath.getName().endsWith(".model")) {
                        // Read the file and load the model
                        J48 loadedModel = (J48) SerializationHelper.read(fs.open(filePath));
                        models.add(loadedModel);
                        context.write(new Text("model_number"), new Text(filePath.toString()));
                    }
                }
            } catch (Exception e) {
                logger.error("Failed to train the model.", e);
                throw new IOException("Failed to train the model.", e);
            }
        }

        @Override
        public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {
            // This method is called for each input record
            // Each record represents a data point with features and a label
            // The label is removed, and the models predict the label using the features
            // The predictions are aggregated to determine the final prediction (mode/max)

            int label_0 = 0;
            int label_1 = 0;
            String trueLabel = null;

            for (Classifier model : models) {
                List<List<String>> allRecords = new ArrayList<>();

                String[] record = String.valueOf(value).split(",");
                List<String> recordList = new ArrayList<>(Arrays.asList(record));
                allRecords.add(recordList);

                trueLabel = recordList.get(recordList.size() - 1);

                // Create attributes for the dataset
                List<Attribute> attributes = new ArrayList<>();
                for (int i = 0; i < allRecords.get(0).size() - 1; i++) {
                    attributes.add(new Attribute("feature" + (i + 1)));
                }

                // Define class labels
                List<String> classLabels = new ArrayList<>();
                classLabels.add("0");
                classLabels.add("1");

                // Add class attribute to the attributes list
                Attribute classAttribute = new Attribute("class", classLabels);
                attributes.add(classAttribute);

                // Create the dataset instance
                Instances dataset = new Instances("Dataset", (ArrayList<Attribute>) attributes, allRecords.size());
                dataset.setClassIndex(attributes.size() - 1);

                // Add records to the dataset
                for (List<String> record_j : allRecords) {
                    weka.core.Instance instance = new DenseInstance(attributes.size());
                    for (int i = 0; i < record_j.size(); i++) {
                        instance.setValue(i, Double.parseDouble(record_j.get(i)));
                    }
                    dataset.add(instance);
                }

                // Make prediction
                String predictedClassLabel = null;
                for (Instance record_i : dataset) {
                    double prediction = 0;
                    try {
                        prediction = model.classifyInstance(record_i);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    predictedClassLabel = String.valueOf(prediction);
                }

                // Increment appropriate counters based on the prediction value
                if (predictedClassLabel.equals("0.0")) {
                    label_0 += 1;
                } else {
                    label_1 += 1;
                }
            }

            // Get the final prediction
            String predictedValue = null;
            if (label_0 >= label_1) {
                predictedValue = "0";
            } else {
                predictedValue = "1";
            }

            // Write the true label and predicted value as output
            //context.write(new Text(trueLabel), new Text(predictedValue));

            // Update accuracy counters
            if (predictedValue.equals("1") && trueLabel.equals("1")) {
                context.getCounter(Accuracy.TRUE_POSITIVE_COUNT).increment(1);

            } else if (predictedValue.equals("0") && trueLabel.equals("0")) {
                context.getCounter(Accuracy.TRUE_NEGATIVE_COUNT).increment(1);

            } else if (predictedValue.equals("1") && trueLabel.equals("0")) {
                context.getCounter(Accuracy.FALSE_POSITIVE_COUNT).increment(1);

            } else if (predictedValue.equals("0") && trueLabel.equals("1")) {
                context.getCounter(Accuracy.FALSE_NEGATIVE_COUNT).increment(1);
            }
        }
    }

    @Override
    public int run(final String[] args) throws Exception {
        final Configuration conf = getConf();
        final Job job = Job.getInstance(conf, "Decision Tree Model Testing");
        job.setJarByClass(ModelTesting.class);
        final Configuration jobConf = job.getConfiguration();
        jobConf.set("mapreduce.output.textoutputformat.separator", "\t");

        job.setMapperClass(TokenizerMapper.class);
        job.setNumReduceTasks(0);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        TextInputFormat.setInputPaths(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));

        // Setup cache file and file system for distributed cache.
        job.addCacheFile(new Path("s3://cs6240-team-ra/input/intermediate_model_output/").toUri());
        job.getConfiguration().set("cache.fs", "s3://cs6240-team-ra/");

        // to run locally and on pseudo, comment the above line and uncomment the below line
        //job.getConfiguration().set("cache.fs", "hdfs://localhost:9870/");

        System.out.println("-------------------------------");
        System.out.println(args[0]);


        int jobCompleted = job.waitForCompletion(true) ? 0 : 1;

        // Calculate precision and recall
        Counter truePos = job.getCounters().findCounter(Accuracy.TRUE_POSITIVE_COUNT);
        Counter trueNeg = job.getCounters().findCounter(Accuracy.TRUE_NEGATIVE_COUNT);
        Counter falsePos = job.getCounters().findCounter(Accuracy.FALSE_POSITIVE_COUNT);
        Counter falseNeg = job.getCounters().findCounter(Accuracy.FALSE_NEGATIVE_COUNT);

        double precision = (double) ((int) truePos.getValue()) / ((int) truePos.getValue() + (int) falsePos.getValue());
        double recall = (double) ((int) truePos.getValue()) / ((int) truePos.getValue() + (int) falseNeg.getValue());

        // Log the accuracy measures
        logger.info("True Positive Count: " + truePos.getValue());
        logger.info("True Negative Count: " + trueNeg.getValue());
        logger.info("False Positive Count: " + falsePos.getValue());
        logger.info("False Negative Count: " + falseNeg.getValue());
        logger.info("-------------------------------");
        logger.info("Precision: " + precision);
        logger.info("Recall: " + recall);
        logger.info("-------------------------------");
        logger.info("Numerator: " + (2 * precision * recall));
        logger.info("Denominator: " + (precision + recall));
        logger.info("-------------------------------");

        // Calculate F1-score
        double f1_score = (2 * precision * recall) / (precision + recall);
        logger.info("The F1-Score of the Ensemble Models is " + f1_score);

        return jobCompleted;

    }

    public static void main(final String[] args) {
        if (args.length != 2) {
            throw new Error("Two arguments required:\n<input-dir> <output-dir>");
        }

        try {
            ToolRunner.run(new ModelTesting(), args);
        } catch (final Exception e) {
            logger.error("", e);
        }
    }

}
