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

/**
 * Parition + braoadcast algorithm.
 * Map-only job
 * Setup: reads models from cache
 * Mapper: makes prediction for each test point
 */
public class ModelTesting extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(ModelTesting.class);

    //saves all the models read from the file cache
    private static List<Classifier> models = new ArrayList<>();

    //global counters for computing precision and recall related parameters
    enum Accuracy {
        TRUE_POSITIVE_COUNT,
        TRUE_NEGATIVE_COUNT,
        FALSE_POSITIVE_COUNT,
        FALSE_NEGATIVE_COUNT
    }
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

        /**
         * The set up reads the models from the file cache and saves them to an array.
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void setup(Context context) throws IOException, InterruptedException {

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
                    }
                }
            } catch (Exception e) {
                logger.error("Failed to train the model.", e);
                throw new IOException("Failed to train the model.", e);
            }
        }

        /**
         * Each map call looks at one test record. For each test record, we loop through
         * all the models are make a prediction using each model. Once that is done, we combine
         * all the predicitons to get one prediction (since there are only 2 labels, we could the occurence
         * of rach label and then pick the one that occured more times than the other). Once we have a prediciton
         * for the test point, we then increment the FP, FN, TP or TN comparing the truw label and the predicted label.
         *
         * @param key is the offset of the test record
         * @param value the test record
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {

            //variables below keep track of the prediction from the ensembles
            int label_0 = 0;
            int label_1 = 0;
            String trueLabel = null;

            //for each test record, loop through all records
            for (Classifier model : models) {
                List<List<String>> allRecords = new ArrayList<>();

                //put record in a list of listd
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
                    label_0 += 1; //if label 0 is predicted
                } else {
                    label_1 += 1; //if label 1 is predicted
                }
            }

            //once we have predicted using all the models, we want a final prediction.
            //so here, the final predicted value is determined by which variables has a higher value.
            // Get the final prediction
            String predictedValue = null;
            if (label_0 >= label_1) {
                predictedValue = "0";
            } else {
                predictedValue = "1";
            }

            // Update accuracy counters.
            // Thiese are global counters. We will update once per test point.
            // We do this so that we can compute the precision, recall and F1 score at the end.
            if (predictedValue.equals("1") && trueLabel.equals("1")) {
                //incremeent true positive count
                context.getCounter(Accuracy.TRUE_POSITIVE_COUNT).increment(1);

            } else if (predictedValue.equals("0") && trueLabel.equals("0")) {
                //incremeent true negative count
                context.getCounter(Accuracy.TRUE_NEGATIVE_COUNT).increment(1);

            } else if (predictedValue.equals("1") && trueLabel.equals("0")) {
                //incremeent false positive count
                context.getCounter(Accuracy.FALSE_POSITIVE_COUNT).increment(1);

            } else if (predictedValue.equals("0") && trueLabel.equals("1")) {
                //incremeent false positive count
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
