package com.pgbde.spark.mllib;

//import com.amazonaws.auth.BasicSessionCredentials;
//import com.amazonaws.auth.InstanceProfileCredentialsProvider;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.apache.spark.sql.functions.col;

//this class is used to Prepare the data for processing.
//Use S3 files and keep it a temp output folder
public class ClickThroughRate {

	public static void main(String[] args) {


		System.out.println("Starting data preparation");
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String inputPath = args[0] ;// "input/";
		String outputPath =args[1] ;// "target/tmp/model";
		String localFlag = (args.length >= 3) ? args[2]:Constants.LOCAL ; //local or not
		String access_key_amazon =  (args.length >= 4) ?args[3]: Constants.DEF_ACCESSKEY ;
		String secret_key_amazon =  (args.length >= 5) ?args[4]: Constants.DEF_SECRETKEY;

// Load the  clickstreamdata.csv -- four attributes - “User ID”, “Song ID”,  “Date” and “Timestamp.”
//5525c71b6213340569f3aa1abc225514,1533115844,_JUdlvPU,20180801
		String path1 = inputPath + Constants.ACTIVITY_FOLDER ;

		SparkSession session = null;
		if(localFlag.equals(Constants.AWS) ) {
			path1 = inputPath+ "/activity/sample100mb.csv"; //Update path to sample 100MB file
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

		//Based on the savemode process the testing data and testing data.

//metadata folder -- “Song ID” to an “Artist ID.”A specific "Song ID" can be related to multiple "Artist IDs".
//Zil3cVnY,516437
			String path2 =inputPath+Constants.METADATA_FOLDER;
//Read the files from the paths.
			Dataset<Row> dataset1 = session.read().format("csv").option("header","false").load(path1);
			Dataset<Row> dataset2 = session.read().format("csv").option("header","false").load(path2);

//		dataset1.show(5);
//		dataset2.show(6);

			dataset1.persist(StorageLevel.MEMORY_AND_DISK_SER());
			dataset2.persist(StorageLevel.MEMORY_AND_DISK_SER());

			System.out.println("Training Data :String indexer started ");
			StringIndexer indexer1 = new StringIndexer().setInputCol("_c0").setOutputCol("userId");
			StringIndexerModel indModel1 = indexer1.fit(dataset1);
			dataset1 = indModel1.transform(dataset1);

//Type cast the columns for better readability with column names.
// Add a rating column for the click event as 1

			Dataset<Row> data1 = dataset1.withColumn("userId",col("userId").cast(DataTypes.IntegerType))
					.withColumn("userId_str", col("_c0").cast(DataTypes.StringType))
					.withColumn("songId", col("_c2").cast(DataTypes.StringType));
			Dataset<Row> data2 = dataset2.withColumn("songId", col("_c0").cast(DataTypes.StringType))
					.withColumn("artistId", col("_c1").cast(DataTypes.IntegerType)) ;

			System.out.println("Training Data :Column updates completed");
//Drop the columns
			String[] drop1Col = {"_c0","_c1","_c2","_c3"};
			for(String col: drop1Col) {
				data1 = data1.drop(col);
				data2 = data2.drop(col);
			}

			System.out.println("Training Data :Column drops completed");
			Dataset<Row> inputData =data1.join(data2,"songId");
			inputData = inputData.na().drop();
			inputData.show(5);
			//inputData.toDF().write().mode(SaveMode.Overwrite).csv(tempPath + Constants.TEMPINPUT_FOLDER);
		//System.out.println("Training Data :Count:" +inputData.count());

		System.out.println("Starting Modelling");

//Set up Ratings object for ALS Model
		JavaRDD<Rating> trainingData = inputData.javaRDD().map(s -> {
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Rating(userId,artistId, new Double(10.0));
		});


// Build the recommendation model using ALS
		JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());
		int rank = 5;
		int numIterations = 10;
		double lambda = 10; //0.01

		System.out.println("Executing ALS training");
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainingData), rank, numIterations, lambda);
		System.out.println("Finished ALS training");

		model.save(jsc.sc(),outputPath+ Constants.MODELOUTPUT_FOLDER);
		System.out.println("Saving the model");

//Notification Clicks -Attributes: "Notification ID", "User ID", and "Date"
//9681,69f1004d6c2395cf556c76498e041d5e,20180826
			String path3 =inputPath+ Constants.NOTIFICATIONCLICK_FOLDER;
//Notification Artists-- "Notification ID" and "Artist ID"
//9553,535722
			String path4 =inputPath+ Constants.NOTIFICATIONACTOR_FOLDER;

			Dataset<Row> dataset3 = session.read().format("csv").option("header","false").load(path3);
			Dataset<Row> dataset4 = session.read().format("csv").option("header","false").load(path4);
//		dataset3.show(7);
//		dataset4.show(8);
			dataset3.persist(StorageLevel.MEMORY_AND_DISK_SER());
			dataset4.persist(StorageLevel.MEMORY_AND_DISK_SER());
			//Convert userId to a integer.
			System.out.println("Testing Data :String indexer started ");
			StringIndexer indexer3 = new StringIndexer().setInputCol("_c1").setOutputCol("userId");
			StringIndexerModel indModel3 = indexer3.fit(dataset3);
			dataset3 = indModel3.transform(dataset3);

			System.out.println("Testing Data :String indexer completed");
//		dataset3.show(7);

			Dataset<Row> data3 = dataset3.withColumn("userId",col("userId").cast(DataTypes.IntegerType))
					.withColumn("notificationId", col("_c0").cast(DataTypes.IntegerType))
					.withColumn("userId_str", col("_c1").cast(DataTypes.StringType))
					//.withColumn("date", col("_c2").cast(DataTypes.IntegerType))
					;
			Dataset<Row> data4 = dataset4.withColumn("notificationId", col("_c0").cast(DataTypes.IntegerType))
					.withColumn("artistId", col("_c1").cast(DataTypes.IntegerType))
					//.withColumn("ratings", functions.lit(0).cast(DataTypes.FloatType))
					;

			System.out.println("Testing Data :Column updates completed");

			for(String col: drop1Col) {
				data3 = data3.drop(col);
				data4 = data4.drop(col);
			}

			System.out.println("Testing Data :Column drops completed");
			Dataset<Row> testingData =data3.join(data4,"notificationId").orderBy(col("artistId"),col("userId"));;
			testingData.show(6);

			//System.out.println("Testing Data :count:" +outputData.count());

//Data analysis
//		JavaRDD<Rating> testingData = outputData.javaRDD().map(s -> {
//			int userId = Integer.parseInt(""+s.get(1));
//			int artistId = Integer.parseInt(""+s.get(3));
//			return new Rating(userId,artistId,0.0);
//		});


		// Evaluate the model on notification data set and get the predictions
		//JavaRDD<Tuple2<Object, Object>> userIdArtistId = testingData.map(r -> new Tuple2<>(r.user(), r.product()));

		JavaRDD<Tuple2<Object, Object>> userIdArtistId = testingData.javaRDD().map(s ->{
			int userId = Integer.parseInt(""+s.get(1));
			int artistId = Integer.parseInt(""+s.get(3));
			return new Tuple2<>(userId, artistId);
		});

		System.out.println("Testing Data :RDD.");
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


		//GroupBy artistId to get predictedUserId and groupby predictedUserIds to generate the clusterId
		Dataset<Row> predictedUserArtistCluster = predictedDataset.groupBy("artistId")
				.agg(functions.collect_list("predictedUserId").alias("predictedUserIds"))
				.groupBy("predictedUserIds")
				.agg(functions.collect_list("artistId").alias("artistIds"))
				.withColumn("clusterId", functions.monotonically_increasing_id()) ;
		predictedUserArtistCluster.show(100);

		System.out.println("Predicted User activity cluster generated.");

		Dataset<Row> testingUserArtistCluster = testingData.groupBy("artistId")
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

}

