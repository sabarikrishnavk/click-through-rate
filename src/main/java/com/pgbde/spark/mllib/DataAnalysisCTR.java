package com.pgbde.spark.mllib;

import org.apache.hadoop.conf.Configuration;
//import com.amazonaws.auth.BasicSessionCredentials;
//import com.amazonaws.auth.InstanceProfileCredentialsProvider;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Function1;
import scala.Tuple2;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import static org.apache.spark.sql.functions.col;


public class DataAnalysisCTR {
	
	public static void main(String[] args) {


		System.out.println("Start Data analysis");
		SparkSession session = null;
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String tempPath = args[0] ;// ""target/tmp/sql"";
		String outputPath =args[1] ;// "target/tmp/model";
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

//		System.out.println("testing data count "+inputCSV.count());
//		System.out.println("training data count "+outputCSV.count());

		Dataset<Row> testingTable = outputCSV.withColumn("notificationId",col("_c0").cast(DataTypes.IntegerType))
				.withColumn("userId", col("_c1").cast(DataTypes.IntegerType))
				.withColumn("userId_str", col("_c2").cast(DataTypes.StringType))
				.withColumn("artistId", col("_c3").cast(DataTypes.IntegerType))
				.drop("_c0")
				.drop("_c1")
				.drop("_c2")
				.drop("_c3")
				.orderBy(col("artistId"),col("userId"));

//		trainingTable.show(100);
//Set up Ratings object for ALS Model
		JavaRDD<Rating> trainingData = inputCSV.javaRDD().map(s -> {
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Rating(userId,artistId, new Double(10.0));
		});

//		outputCSV.show(4);
		System.out.println("Successfully created training RDD.");
		JavaRDD<Rating> testingData = outputCSV.javaRDD().map(s -> {
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Rating(userId,artistId,0.0);
		});
		System.out.println("Successfully created testing RDD.");


// Build the recommendation model using ALS
		JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());
		int rank = 5;
		int numIterations = 10;
		double lambda = 10; //0.01

		System.out.println("Executing ALS training");
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainingData), rank, numIterations, lambda);
		System.out.println("Finished ALS training");


		// Evaluate the model on notification data set and get the predictions
		JavaRDD<Tuple2<Object, Object>> userIdArtistId = testingData.map(r -> new Tuple2<>(r.user(), r.product()));


		JavaRDD<Rating> rdd = model.predict(JavaRDD.toRDD(userIdArtistId)).toJavaRDD();

		
		StructType schema = DataTypes.createStructType(new StructField[] {
				DataTypes.createStructField("artistId",  DataTypes.IntegerType, true),
				DataTypes.createStructField("predictedUserId", DataTypes.IntegerType, true),
				DataTypes.createStructField("rating", DataTypes.DoubleType, true) 
		});
		JavaRDD<Row> rowRDD = rdd.flatMap(
			new FlatMapFunction<Rating, Row>() {
				 private static final long serialVersionUID = 5481855142090322683L;
				 @Override
				 public Iterator<Row> call(Rating r) throws Exception {
					 List<Row> list = new ArrayList<>();
					 list.add(RowFactory.create(r.product() ,r.user(),r.rating()));
					 return list.iterator();
				 }
		 });
		Dataset<Row> predictedDataset = session.sqlContext().createDataFrame(rowRDD, schema).toDF().orderBy(col("artistId"),col("predictedUserId"));
//		dataset.show(100);
		//Since n Saavn, the notifications are pushed with the intention of notifying users about their preferred artists.
		// In other words, notification informs users about the updates from their favoured artists.
		//Each cluster is targeted with information about only one particular artist.
		// Simply put, each group should be associated with only one artist and,
		// just the notifications related to that specific artist must be pushed to the users within that cluster.
		// notificationId is grouped by artistId and userId
		Dataset<Row> predictedCnt = predictedDataset//.filter("rating >0 ")
				.groupBy("artistId")
				.agg(functions.count("predictedUserId"))
				.toDF( "artistId","predictedUserIdCnt");
//		predictedCnt.show(100);

		//Get the artistId group from notification testing data set
		Dataset<Row> testingCnt= testingTable
				.groupBy("artistId")
				.agg(functions.count("userId"))
				.toDF("artistId","clickedUserIdCnt");
//		trainingCnt.show(100);


		//GroupBy artistId to get predictedUserId and groupby predictedUserIds to generate the clusterId
		Dataset<Row> predictedUserArtistCluster = predictedDataset.groupBy("artistId")
				.agg(functions.collect_list("predictedUserId").alias("predictedUserIds"))
				.groupBy("predictedUserIds")
				.agg(functions.collect_list("artistId").alias("artistIds"))
				.withColumn("clusterId", functions.monotonically_increasing_id()) ;
		predictedUserArtistCluster.show(100);
		System.out.println("Predicted User activity cluster generated.");

		Dataset<Row> testingUserArtistCluster = testingTable.groupBy("artistId")
				.agg(functions.collect_list("userId").alias("testUserIds"))
				.groupBy("testUserIds")
				.agg(functions.collect_list("artistId").alias("artistIds"))
				.withColumn("clusterId", functions.monotonically_increasing_id()) ;
		testingUserArtistCluster.show(100);

		//userArtistCluster.toDF().write().mode(SaveMode.Overwrite).csv(outputPath+ Constants.USERACTIVITY_FOLDER);
		System.out.println("Test User activity cluster generated.");

		//Click-Through Rate(CTR) is a measure that indicates
		// the ratio of the number of users who clicked on the pushed notification (clickedUserIdCnt from training data set)
		// to the total number of users to whom that notification was pushed.(predictedUserIdCnt from predicted data set)

		Dataset<Row> finalTable = testingUserArtistCluster.join(predictedUserArtistCluster,"clusterId")
				.select( col("clusterId"),
						functions.size(col("testUserIds")),
						functions.size(col("predictedUserIds"))
						)
				.toDF("clusterId","clickedUserIdCnt" ,"predictedUserIdCnt")
				.selectExpr("clusterId" ,"clickedUserIdCnt" ,"predictedUserIdCnt", "clickedUserIdCnt / predictedUserIdCnt as ctr")
				.orderBy(col("ctr").desc());

		finalTable.toDF().write().mode(SaveMode.Overwrite).csv(outputPath+ Constants.CTR_FOLDER);
		System.out.println("Click through rate generated.");

		session.close();
	}

//		JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
//				model.predict(JavaRDD.toRDD(userIdArtistId)).toJavaRDD()
//						.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
//		);
//
//		//Join the predicted ratings and actual values
//		JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(
//				trainingData.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())))
//				.join(predictions).values();
//
//		//calculate the mean square error.
//		double MSE = ratesAndPreds.mapToDouble(pair -> {
//			double err = pair._1() - pair._2();
//			return err * err;
//		}).mean();
//
//
//		System.out.println("Mean Squared Error = " + MSE);
//
//		// Save and load model
//		model.save(jsc.sc(), outputPath);
//		MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),outputPath);

}

