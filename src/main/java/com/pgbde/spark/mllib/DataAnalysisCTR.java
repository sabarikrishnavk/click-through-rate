package com.pgbde.spark.mllib;

import org.apache.hadoop.conf.Configuration;
//import com.amazonaws.auth.BasicSessionCredentials;
//import com.amazonaws.auth.InstanceProfileCredentialsProvider;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;


import static org.apache.spark.sql.functions.col;


public class DataAnalysisCTR {
	
	public static void main(String[] args) {


		System.out.println("Start Data analysis");
		SparkSession session = null;
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String tempPath = args[0] ;// ""target/tmp/sql"";
		String outputPath =args[1] ;// "output";
		String localFlag = (args.length >= 3) ? args[2]:"local" ; //local or not
		String access_key_amazon =  (args.length >= 4) ?args[3]:"accesskey";
		String secret_key_amazon =  (args.length >= 5) ?args[4]:"secretkey";


		String path1 = tempPath + Constants.TEMPINPUT_FOLDER ;

		String path2 =tempPath+Constants.TEMPOUTPUT_FOLDER;

		if(localFlag.equals(Constants.AWS) ) {
			session = SparkSession.builder()
					.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
					.config("spark.hadoop.fs.s3a.access.key", access_key_amazon)
					.config("spark.hadoop.fs.s3a.secret.key", secret_key_amazon)
					.config("fs.s3a.connection.ssl.enabled", "false")
					.config("spark.network.timeout", "600s")
					.config("spark.executor.heartbeatInterval", "500s")
					.getOrCreate();
		}else{
			session = SparkSession.builder().master("local")
					.config("fs.s3a.connection.ssl.enabled", "false")
					.config("spark.network.timeout", "600s")
					.config("spark.executor.heartbeatInterval", "500s")
					.getOrCreate();
		}

//Read the files from the paths.
		Dataset<Row> inputCSV = session.read().format("csv").option("header","false").load(path1);
		Dataset<Row> outputCSV = session.read().format("csv").option("header","false").load(path2);



//		Dataset<Row> inputData = inputCSV.withColumn("songId",col("_c0").cast(DataTypes.StringType))
//				.withColumn("userId", col("_c1").cast(DataTypes.IntegerType))
//				.withColumn("userId_str", col("_c2").cast(DataTypes.StringType))
//				.withColumn("artistId", col("_c3").cast(DataTypes.IntegerType));
//
//		Dataset<Row> outputData = outputCSV.withColumn("notificationId",col("_c0").cast(DataTypes.IntegerType))
//				.withColumn("userId", col("_c1").cast(DataTypes.IntegerType))
//				.withColumn("userId_str", col("_c2").cast(DataTypes.StringType))
//				.withColumn("artistId", col("_c3").cast(DataTypes.IntegerType));


		inputCSV.show(10);
		outputCSV.show(10);
//Set up Ratings object for ALS Model
		JavaRDD<Rating> trainingData = inputCSV.javaRDD().map(s -> {
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Rating(userId,artistId, new Double(1.0));
		});

		System.out.println("Successfully created training RDD.");
		JavaRDD<Rating> testingData = outputCSV.javaRDD().map(s -> {
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Rating(userId,artistId,0.0);
		});
		System.out.println("Successfully created testing RDD.");

		trainingData.take(10).parallelStream().forEach(rating -> {
			System.out.println(rating);
		});
		System.out.println("---------------.");
		testingData.take(10).parallelStream().forEach(rating -> {
			System.out.println(rating);
		});

// Build the recommendation model using ALS
		JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());
		int rank = 10;
		int numIterations = 10;

		System.out.println("Executing ALS training");
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainingData), rank, numIterations, 0.01);
		System.out.println("Finished ALS training");


		// Evaluate the model on notification data set and get the predictions
		JavaRDD<Tuple2<Object, Object>> userIdArtistId = testingData.map(r -> new Tuple2<>(r.user(), r.product()));

		JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
				model.predict(JavaRDD.toRDD(userIdArtistId)).toJavaRDD()
						.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
		);

		//Join the predicted ratings and actual values
		JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(
				trainingData.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())))
				.join(predictions).values();

		//calculate the mean square error.
		double MSE = ratesAndPreds.mapToDouble(pair -> {
			double err = pair._1() - pair._2();
			return err * err;
		}).mean();


		System.out.println("Mean Squared Error = " + MSE);

		// Save and load model
		model.save(jsc.sc(), outputPath);
		MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),outputPath);

		session.close();
	}

}

