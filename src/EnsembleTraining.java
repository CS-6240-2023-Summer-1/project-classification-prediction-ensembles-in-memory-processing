package Decision_Tree;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.hadoop.io.NullWritable;


public class EnsembleTraining extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(DT.class);

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);
        private final Text word = new Text();

        @Override
        public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {
            final StringTokenizer itr = new StringTokenizer(value.toString(), ",");
            //key variable contains a random number between 0 and 100.
            int start_range = 0;
            int end_range = 100;
            int key = random.nextInt(end_range - start_range + 1) + start_range;
            //for each input record, we assign the key so that the that input record goes to that reduce task
            //value is the entire record (features + record) from the train set
            context.write(new Text(key.toString()), value);
        }
    }


    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
        private final IntWritable result = new IntWritable();

        @Override
        public void reduce(final Text key, final Iterable<Text> values, final Context context) throws IOException, InterruptedException {
            List<List<String>> allRecords = new ArrayList<>();

            for (final Text val : values) {
                String[] record = val.split(","); // Split the text into an array using the comma delimiter
                List<String> recordList = Arrays.asList(record);
                allRecords.add(recordList);
            }

            // Create a list of attributes/features
            List<Attribute> attributes = new ArrayList<>();
            for (int i = 0; i < allRecords.get(0).size() - 1; i++) {
                attributes.add(new Attribute("feature" + (i + 1)));
            }
            attributes.add(new Attribute("class")); //to add class label

            // Create an empty Instances object
            Instances dataset = new Instances("Dataset", attributes, allRecords.size());

            // Add the data records to the Instances object
            for (List<String> record : allRecords) {
                Instance instance = new DenseInstance(attributes.size());

                for (int i = 0; i < record.size(); i++) {
                    if (i == record.size() - 1) {
                        instance.setValue(i, record.get(i)); // Set class value
                    } else {
                        instance.setValue(i, Double.parseDouble(record.get(i))); // Set feature value
                    }
                }

                dataset.add(instance);
            }

            // Set the class index
            dataset.setClassIndex(attributes.size() - 1);


            //each key has a set of records associated with it
            //we will train all of the records
            //we want to seperate X and Y.

            //train the model on X.
            //save model.
    }

    @Override
    public int run(final String[] args) throws Exception {
        final Configuration conf = getConf();
        final Job job = Job.getInstance(conf, "Label Count");
        job.setJarByClass(DT.class);

        // Set input format class to handle gzipped CSV files
        job.setInputFormatClass(TextInputFormat.class);

        final Configuration jobConf = job.getConfiguration();
        jobConf.set("mapreduce.output.textoutputformat.separator", "\t");

        // Delete output directory, only to ease local development; will not work on AWS.
        // You may uncomment and modify this part according to your requirements.
//        final FileSystem fileSystem = FileSystem.get(conf);
//        if (fileSystem.exists(new Path(args[1]))) {
//            fileSystem.delete(new Path(args[1]), true);
//        }

        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(final String[] args) {
        if (args.length != 2) {
            throw new Error("Two arguments required:\n<input-dir> <output-dir>");
        }

        try {
            ToolRunner.run(new DT(), args);
        } catch (final Exception e) {
            logger.error("", e);
        }
    }

}
