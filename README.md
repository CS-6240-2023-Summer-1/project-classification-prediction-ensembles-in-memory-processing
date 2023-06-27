# Ensemble Classification & Prediction for In-Memory Processing 
-----------
## Ideation:
Train an ensemble classification or prediction model consisting of individual models from an existing library designed for in-memory processing on a single machine. You can use such libraries for training and prediction of individual models, but you must code from scratch the parallel framework for ensemble training and prediction, and for efficient exploration of the best model parameters. Then use your ensemble model to make predictions and report its accuracy. You
may apply this solution to a Kaggle competition.

CS6240 Final Project
Summer 2023

Author
-----------
- Rutuja Shah
- Ashir Mehta

Installation
------------
These components are installed:
- OpenJDK 11
- Hadoop 3.3.5
- Maven (Tested with version 3.6.3)
- AWS CLI (Tested with version 1.22.34)

- Scala 2.12.17 (you can install this specific version with the Coursier CLI tool which also needs to be installed)
- Spark 3.3.2 (without bundled Hadoop)

After downloading the hadoop and spark installations, move them to an appropriate directory:

`mv hadoop-3.3.5 /usr/local/hadoop-3.3.5`

`mv spark-3.3.2-bin-without-hadoop /usr/local/spark-3.3.2-bin-without-hadoop`

Environment
-----------
1) Example ~/.bash_aliases:
	```
	export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
	export HADOOP_HOME=/usr/local/hadoop-3.3.5
	export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
	export SCALA_HOME=/usr/share/scala
	export SPARK_HOME=/usr/local/spark-3.3.2-bin-without-hadoop
	export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SCALA_HOME/bin:$SPARK_HOME/bin
	export SPARK_DIST_CLASSPATH=$(hadoop classpath)
	```

2) Explicitly set `JAVA_HOME` in `$HADOOP_HOME/etc/hadoop/hadoop-env.sh`:

	`export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`

Execution
---------
All of the build & execution commands are organized in the Makefile.
1) Unzip project file.
2) Open command prompt.
3) Navigate to directory where project files unzipped.
4) Edit the Makefile to customize the environment at the top.
	Sufficient for standalone: hadoop.root, jar.name, local.input
	Other defaults acceptable for running standalone.
5) Standalone Hadoop:
	- `make switch-standalone`		-- set standalone Hadoop environment (execute once)
	- `make local`
6) Pseudo-Distributed Hadoop: (https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation)
	- `make switch-pseudo`			-- set pseudo-clustered Hadoop environment (execute once)
	- `make pseudo`					-- first execution
	- `make pseudoq`				-- later executions since namenode and datanode already running 
7) AWS EMR Hadoop: (you must configure the emr.* config parameters at top of Makefile)
	- `make make-bucket`			-- only before first execution
	- `make upload-input-aws`		-- only before first execution
	- `make aws`					-- check for successful execution with web interface (aws.amazon.com)
	- `download-output-aws`		-- after successful execution & termination


Files Description
---------
1. TrainTest.java: This Java code performs a train-test split on a dataset stored in a CSV file. It reads the dataset from the file, shuffles the rows randomly, and divides them into a training set and a test set based on a specified ratio. The program then creates separate folders for the train and test datasets if they don't already exist. It saves the train and test datasets into separate CSV files within their respective folders. Finally, it outputs the sizes of the train and test sets.

2. EnsembleTraining.java: This Java code implements ensemble training using the Hadoop MapReduce framework and the Weka machine learning library. It uses the train test split dataset and trains individual decision tree classifiers on each subset using the MapReduce. The trained models are serialized and saved into the intermediate output model folder so that it can be uploaded into the File Cache for Model Testing. It utilizes distributed computing to train the models in parallel, improving efficiency.

3. ModelTesting.java:
