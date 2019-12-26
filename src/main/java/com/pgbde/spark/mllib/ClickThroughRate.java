package com.pgbde.spark.mllib;

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
import scala.Tuple2;

import static org.apache.spark.sql.functions.col;


public class ClickThroughRate {
	
	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String inputPath = args[0] ;// "input/";
		String outputPath =args[1] ;// "output";

// Load the  clickstreamdata.csv -- four attributes - “User ID”, “Song ID”,  “Date” and “Timestamp.”
//5525c71b6213340569f3aa1abc225514,1533115844,_JUdlvPU,20180801
		String path1 = inputPath+ "/activity/sample100mb.csv";

//metadata folder -- “Song ID” to an “Artist ID.”A specific "Song ID" can be related to multiple "Artist IDs".
//Zil3cVnY,516437
		String path2 =inputPath+"/newmetadata";

//Notification Clicks -Attributes: "Notification ID", "User ID", and "Date"
//9681,69f1004d6c2395cf556c76498e041d5e,20180826
		String path3 =inputPath+"/notification_clicks";
//Notification Artists-- "Notification ID" and "Artist ID"
//9553,535722
		String path4 =inputPath+"/notification_actor";


		SparkSession session = SparkSession.builder().master("local")
				.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
				//.config("spark.hadoop.fs.s3a.access.key", access_key_amazon)
				//.config("spark.hadoop.fs.s3a.secret.key", secret_key_amazon)
				.config("fs.s3a.connection.ssl.enabled", "false").config("spark.network.timeout", "600s").config("spark.executor.heartbeatInterval", "500s")
				.getOrCreate();

//Read the files from the paths.
		Dataset<Row> dataset1 = session.read().format("csv").option("header","false").load(path1);
		Dataset<Row> dataset2 = session.read().format("csv").option("header","false").load(path2);
		Dataset<Row> dataset3 = session.read().format("csv").option("header","false").load(path3);
		Dataset<Row> dataset4 = session.read().format("csv").option("header","false").load(path4);

//		System.out.println(dataset1.schema());
//		System.out.println(dataset2.schema());

//Convert userId to a integer.
		StringIndexer indexer = new StringIndexer().setInputCol("_c0").setOutputCol("userId");
		StringIndexerModel indModel1 = indexer.fit(dataset1);
		dataset1 = indModel1.transform(dataset1);

		StringIndexerModel indModel3 = indexer.fit(dataset3);
		dataset3 = indModel3.transform(dataset3);

//Type cast the columns for better readability with column names.
// Add a rating column for the click event as 1

		Dataset<Row> data1 = dataset1.withColumn("userId",col("userId").cast(DataTypes.IntegerType))
				.withColumn("userId_str", col("_c0").cast(DataTypes.StringType))
				//.withColumn("timestamp", col("_c1").cast(DataTypes.IntegerType))
				//.withColumn("date", col("_c3").cast(DataTypes.IntegerType))
				.withColumn("songId", col("_c2").cast(DataTypes.StringType))
				.withColumn("ratings", functions.lit(1).cast(DataTypes.FloatType));
		Dataset<Row> data2 = dataset2.withColumn("songId", col("_c0").cast(DataTypes.StringType))
				.withColumn("artistId", col("_c1").cast(DataTypes.IntegerType)) ;

		Dataset<Row> data3 = dataset3.withColumn("userId",col("userId").cast(DataTypes.IntegerType))
				.withColumn("notificationId", col("_c0").cast(DataTypes.IntegerType))
				.withColumn("userId_str", col("_c1").cast(DataTypes.StringType))
				//.withColumn("date", col("_c2").cast(DataTypes.IntegerType))
				;
		Dataset<Row> data4 = dataset4.withColumn("notificationId", col("_c0").cast(DataTypes.IntegerType))
				.withColumn("artistId", col("_c1").cast(DataTypes.IntegerType))
				//.withColumn("ratings", functions.lit(0).cast(DataTypes.FloatType))
				;

//Drop the columns
		String[] drop1Col = {"_c0","_c1","_c2","_c3"};
		for(String col: drop1Col) {
			data1 = data1.drop(col);
			data2 = data2.drop(col);
			data3 = data3.drop(col);
			data4 = data4.drop(col);
		}
//		data1.show(10);
//		data2.show(10);

//Join the dataset to create a single
		Dataset<Row> inputData =data1.join(data2,"songId");
		inputData = inputData.na().drop();
		Dataset<Row> outputData =data3.join(data4,"notificationId");

		inputData = inputData.drop("songId");
		outputData = outputData.drop("notificationId");

		inputData.show(20);
		outputData.show(20);

		System.out.println("Successfully completed joining the tables.");

//Set up Ratings object for ALS ModeL
		//JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());
		JavaRDD<Row> inputRDD = inputData.javaRDD();
		JavaRDD<Rating> trainingData = inputRDD.map(s -> {
			return new Rating(s.getInt(0),s.getInt(3),
					Double.parseDouble(""+s.getFloat(2)));
		});

		JavaRDD<Row> outputRDD = outputData.javaRDD();
		JavaRDD<Rating> testingData = outputRDD.map(s -> {
			return new Rating(s.getInt(0),s.getInt(2),0.0);
		});
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

