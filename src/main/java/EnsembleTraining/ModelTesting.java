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

public class ModelTesting extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(EnsembleTraining.class);


    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = new Configuration();
            Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);

            for (Path cacheFile : cacheFiles) {
                // Load the model from the .model file
                Classifier classifier = loadModel(cacheFile.toString());
            }


            @Override
        public void map(final Object key, final Text value, final Context context) throws java.io.IOException, InterruptedException {

        }

    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(final Text key, final Iterable<Text> values, final Context context) throws java.io.IOException, InterruptedException {


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
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Setup cache file and file system for distributed cache.
        job.addCacheFile(new Path(args[0] + "/intermediate_model_output").toUri());
        job.getConfiguration().set("cache.fs", "s3://cs6240-team-ra/");

        //do the below if you want to run locally.
        //job.getConfiguration().set("cache.fs", "hdfs://localhost:9000/");

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
