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
import weka.core.SerializationHelper;

public class ModelTesting extends Configured implements Tool {
    private static final Logger logger = LogManager.getLogger(ModelTesting.class);

    private static List<Classifier> models = new ArrayList<>();
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

        }
    }

    @Override
    public int run(final String[] args) throws Exception {
        final Configuration conf = getConf();
        final Job job = Job.getInstance(conf, "Decision Tree");
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
        job.addCacheFile(new Path(args[0] + "/intermediate_model_output/").toUri());
        job.getConfiguration().set("cache.fs", "s3://cs6240-team-ra/");

        // to run locally and on pseudo, comment the above line and uncomment the below line
        //job.getConfiguration().set("cache.fs", "hdfs://localhost:9870/");

        System.out.println("-------------------------------");

        return job.waitForCompletion(true) ? 0 : 1;
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
