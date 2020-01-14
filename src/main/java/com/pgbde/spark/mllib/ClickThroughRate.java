package com.pgbde.spark.mllib;

//import com.amazonaws.auth.BasicSessionCredentials;
//import com.amazonaws.auth.InstanceProfileCredentialsProvider;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.PairRDDFunctions;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import scala.Function1;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.apache.spark.sql.functions.col;

//this class is used to Prepare the data for processing.
//Use S3 files and keep it a temp output folder
public class ClickThroughRate {

	public static void main(String[] args) {


		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String inputPath = args[0] ;// "input/";
		String outputPath =args[1] ;// "target/tmp/model";
		String localFlag = (args.length >= 3) ? args[2]:Constants.LOCAL ; //local or not
		String access_key_amazon =  (args.length >= 4) ?args[3]: Constants.DEF_ACCESSKEY ;
		String secret_key_amazon =  (args.length >= 5) ?args[4]: Constants.DEF_SECRETKEY;
		int startPoint =  (args.length >= 6) ?Integer.parseInt(args[5]): Constants.PROGRAM_CHECKPOINT_1;

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
					//.config("spark.sql.shuffle.partitions","200")
					.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
					.config("spark.sql.autoBroadcastJoinThreshold", -1)
					.getOrCreate();
		}else{
			session = SparkSession.builder().master("local")
					.config("fs.s3a.connection.ssl.enabled", "false")
					.config("spark.network.timeout", "600s")
					.config("spark.executor.heartbeatInterval", "500s")
					.config("spark.sql.shuffle.partitions","200")
					.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
					.getOrCreate();
		}

// Build the recommendation model using ALS
		JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());
		int rank = 5;
		int numIterations = 10;
		double lambda = 10; //0.01

		//Based on the starting point run the data preparation and modeling using training data set
		// or load the model and execute the prediction using testing data set.
		System.out.println("Starting point "+startPoint);
//Read the files from the paths.
		if(startPoint <= Constants.PROGRAM_CHECKPOINT_1) {
			System.out.println("Starting data preparation and modelling");
			//“User ID”, “Song ID”,  “Date” and “Timestamp.”
			Dataset<Row> dataset1 = session.read()
					.format("csv").option("header", "false")
					.load(path1)
					.toDF("userId_str","date","songId","timestamp") ;

//metadata folder -- “Song ID” to an “Artist ID.”A specific "Song ID" can be related to multiple "Artist IDs".
//Zil3cVnY,516437
			String path2 = inputPath + Constants.METADATA_FOLDER;
			Dataset<Row> dataset2 = session.read()
					.format("csv").option("header", "false")
					.load(path2)
					.toDF("songId","artistId")
					.withColumn("artistId", col("artistId").cast(DataTypes.IntegerType)) ;

//			dataset1.show(5);
//			dataset2.show(6);

//			dataset1.persist(StorageLevel.MEMORY_AND_DISK_SER());
//			dataset2.persist(StorageLevel.MEMORY_AND_DISK_SER());

			System.out.println("Training Data :String indexer started ");
			StringIndexer indexer1 = new StringIndexer().setInputCol("userId_str").setOutputCol("userId");
			StringIndexerModel indModel1 = indexer1.fit(dataset1);
			dataset1 = indModel1.transform(dataset1).withColumn("userId", col("userId").cast(DataTypes.IntegerType));
			System.out.println("Training Data :String indexer completed");

			Dataset<Row> inputData = dataset1.join(dataset2, "songId").na().drop();
//			inputData.show(5);

			System.out.println("Starting Modelling");
//Set up Ratings object for ALS Model
			JavaRDD<Rating> trainingData = inputData.javaRDD().map(s -> {
				int userId = Integer.parseInt("" + s.get(4));
				int artistId = Integer.parseInt("" + s.get(5));
				return new Rating(userId, artistId, new Double(1.0));
			});

			System.out.println("Executing ALS training");
			MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainingData), rank, numIterations, lambda);
			System.out.println("Finished ALS training");

			model.save(jsc.sc(), outputPath + Constants.MODELOUTPUT_FOLDER);
			System.out.println("Saving the model");
		}

		if(startPoint <= Constants.PROGRAM_CHECKPOINT_2) {
			System.out.println("Starting data analysis and predictions");
//Notification Clicks -Attributes: "Notification ID", "User ID", and "Date"
//9681,69f1004d6c2395cf556c76498e041d5e,20180826
			String path3 = inputPath + Constants.NOTIFICATIONCLICK_FOLDER;
			Dataset<Row> dataset3 = session.read()
					.format("csv").option("header", "false").load(path3)
					.toDF("notificationId", "userId_str", "date");
//Notification Artists-- "Notification ID" and "Artist ID"
//9553,535722
			String path4 = inputPath + Constants.NOTIFICATIONACTOR_FOLDER;
			Dataset<Row> dataset4 = session.read().format("csv")
					.option("header", "false")
					.load(path4)
					.toDF("notificationId", "artistId")
					.withColumn("artistId", col("artistId").cast(DataTypes.IntegerType));

//			dataset3.show(7);
//			dataset4.show(8);
//			dataset3.persist(StorageLevel.MEMORY_AND_DISK_SER());
			dataset4.persist(StorageLevel.MEMORY_AND_DISK_SER());

			//Convert userId to a integer.
			System.out.println("Testing Data :String indexer started ");
			StringIndexer indexer3 = new StringIndexer().setInputCol("userId_str").setOutputCol("userId");
			StringIndexerModel indModel3 = indexer3.fit(dataset3);
			dataset3 = indModel3.transform(dataset3).withColumn("userId", col("userId").cast(DataTypes.IntegerType));
			System.out.println("Testing Data :String indexer completed");

			Dataset<Row> testingData = dataset3
					.join(dataset4, "notificationId")
//					.drop(col("notificationId"))
					.drop(col("date"))
					.drop(col("userId_str"))
					.dropDuplicates()
					.orderBy(col("artistId"), col("userId"));
			//System.out.println("Testing Data :clickData count." + clickData.count());
			testingData.persist(StorageLevel.MEMORY_AND_DISK_SER());
			System.out.println("Testing Data : Clean up completed");

			JavaRDD<Tuple2<Object, Object>> userIdArtistId = testingData.javaRDD().map(s -> {
				int userId = Integer.parseInt("" + s.get(1));
				int artistId = Integer.parseInt("" + s.get(2));
				return new Tuple2<>(userId, artistId);
			});
			System.out.println("Prediction :Testing Data :RDD.");

			//Load the model using spark context.
			MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(), outputPath + Constants.MODELOUTPUT_FOLDER);

			JavaRDD<Rating> rdd = sameModel.predict(userIdArtistId.rdd()).toJavaRDD();

			System.out.println("Prediction :Loaded model.");
			StructType schema = DataTypes.createStructType(new StructField[]{
					DataTypes.createStructField("artistId", DataTypes.IntegerType, true),
					DataTypes.createStructField("predictedUserId", DataTypes.IntegerType, true),
					DataTypes.createStructField("rating", DataTypes.DoubleType, true)
			});
			JavaRDD<Row> rowRDD = rdd.flatMap(
					new FlatMapFunction<Rating, Row>() {
						private static final long serialVersionUID = 5481855142090322683L;

						@Override
						public Iterator<Row> call(Rating r) throws Exception {
							List<Row> list = new ArrayList<>();
							list.add(RowFactory.create(r.product(), r.user(), r.rating()));
							return list.iterator();
						}
					});
			Dataset<Row> predictedDataset = session.sqlContext()
					.createDataFrame(rowRDD, schema)
					.toDF()
					.orderBy(col("artistId"), col("predictedUserId"));

			System.out.println("Prediction : Dataset creation completed");
//			predictedDataset.persist(StorageLevel.MEMORY_AND_DISK_SER());

//		dataset.show(100);
			//Since n Saavn, the notifications are pushed with the intention of notifying users about their preferred artists.
			// In other words, notification informs users about the updates from their favoured artists.
			//Each cluster is targeted with information about only one particular artist.
			// Simply put, each group should be associated with only one artist and,
			// just the notifications related to that specific artist must be pushed to the users within that cluster.

			//GroupBy artistId to get predictedUserId and groupby predictedUserIds to generate the clusterId
			Dataset<Row> predictedUserArtistCluster = predictedDataset.groupBy("artistId")
					.agg(functions.collect_list("predictedUserId").alias("predictedUserIds"))
					.withColumn("clusterId", functions.monotonically_increasing_id());
			predictedUserArtistCluster.show(100);


			System.out.println("Predicted User activity cluster generated.");

			Dataset<Row> testingUserArtistCluster = testingData.groupBy( "artistId")
					.agg(functions.collect_list("userId").alias("testUserIds"))
					.withColumn("clusterId", functions.monotonically_increasing_id());

			testingUserArtistCluster= testingUserArtistCluster.join(dataset4,"artistId");
			testingUserArtistCluster.show(100);

			//userArtistCluster.toDF().write().mode(SaveMode.Overwrite).csv(outputPath+ Constants.USERACTIVITY_FOLDER);
			System.out.println("Test User activity cluster generated.");

			//Click-Through Rate(CTR) is a measure that indicates
			// the ratio of the number of users who clicked on the pushed notification (clickedUserIdCnt from training data set)
			// to the total number of users to whom that notification was pushed.(predictedUserIdCnt from predicted data set)

			Dataset<Row> finalTable = testingUserArtistCluster.join(predictedUserArtistCluster, "clusterId")
					.select(col("notificationId"),col("clusterId"),
							functions.size(col("testUserIds")),
							functions.size(col("predictedUserIds"))
					)
					.toDF("notificationId","clusterId", "clickedUserIdCnt", "predictedUserIdCnt")
					.selectExpr("notificationId","clusterId", "clickedUserIdCnt", "predictedUserIdCnt", "clickedUserIdCnt / predictedUserIdCnt as ctr")
					.orderBy(col("ctr").desc());

			finalTable.toDF().write().mode(SaveMode.Overwrite).csv(outputPath + Constants.CTR_FOLDER);
			System.out.println("Click through rate generated.");


		}

		session.close();
	}

}

