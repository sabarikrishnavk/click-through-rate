package com.pgbde.spark.mllib;

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
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import static org.apache.spark.sql.functions.col;

//this class is used to Prepare the data for processing.
//Use S3 files and keep it a temp output folder
public class DataPreparationCTR {
	
	public static void main(String[] args) {


		System.out.println("Starting data preparation");
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		String inputPath = args[0] ;// "input/";
		String tempPath =args[1] ;// ""target/tmp/sql"";
		String localFlag = (args.length >= 3) ? args[2]:Constants.LOCAL ; //local or not
		String access_key_amazon =  (args.length >= 4) ?args[3]: Constants.DEF_ACCESSKEY ;
		String secret_key_amazon =  (args.length >= 5) ?args[4]: Constants.DEF_SECRETKEY;

// Load the  clickstreamdata.csv -- four attributes - “User ID”, “Song ID”,  “Date” and “Timestamp.”
//5525c71b6213340569f3aa1abc225514,1533115844,_JUdlvPU,20180801
		String path1 = inputPath + Constants.ACTIVITY_FOLDER ;

//metadata folder -- “Song ID” to an “Artist ID.”A specific "Song ID" can be related to multiple "Artist IDs".
//Zil3cVnY,516437
		String path2 =inputPath+Constants.METADATA_FOLDER;

//Notification Clicks -Attributes: "Notification ID", "User ID", and "Date"
//9681,69f1004d6c2395cf556c76498e041d5e,20180826
		String path3 =inputPath+ Constants.NOTIFICATIONCLICK_FOLDER;
//Notification Artists-- "Notification ID" and "Artist ID"
//9553,535722
		String path4 =inputPath+ Constants.NOTIFICATIONACTOR_FOLDER;

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




//Read the files from the paths.
		Dataset<Row> dataset1 = session.read().format("csv").option("header","false").load(path1);
		Dataset<Row> dataset2 = session.read().format("csv").option("header","false").load(path2);
		Dataset<Row> dataset3 = session.read().format("csv").option("header","false").load(path3);
		Dataset<Row> dataset4 = session.read().format("csv").option("header","false").load(path4);

//		dataset1.show(5);
//		dataset2.show(6);
//		dataset3.show(7);
//		dataset4.show(8);

		dataset1.persist(StorageLevel.MEMORY_AND_DISK_SER());
		dataset2.persist(StorageLevel.MEMORY_AND_DISK_SER());
		dataset3.persist(StorageLevel.MEMORY_AND_DISK_SER());
		dataset4.persist(StorageLevel.MEMORY_AND_DISK_SER());

		System.out.println("String indexer for testing data set started :1 ");
		StringIndexer indexer1 = new StringIndexer().setInputCol("_c0").setOutputCol("userId");
		StringIndexerModel indModel1 = indexer1.fit(dataset1);
		dataset1 = indModel1.transform(dataset1);


//		dataset1.show(5);

		//Convert userId to a integer.
		System.out.println("String indexer for training data set started :2 ");
		StringIndexer indexer3 = new StringIndexer().setInputCol("_c1").setOutputCol("userId");
		StringIndexerModel indModel3 = indexer3.fit(dataset3);
		dataset3 = indModel3.transform(dataset3);

		System.out.println("String indexer completed");
//		dataset3.show(7);

//Type cast the columns for better readability with column names.
// Add a rating column for the click event as 1

		Dataset<Row> data1 = dataset1.withColumn("userId",col("userId").cast(DataTypes.IntegerType))
				.withColumn("userId_str", col("_c0").cast(DataTypes.StringType))
				.withColumn("songId", col("_c2").cast(DataTypes.StringType));
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

		System.out.println("Column updates completed");
//Drop the columns
		String[] drop1Col = {"_c0","_c1","_c2","_c3"};
		for(String col: drop1Col) {
			data1 = data1.drop(col);
			data2 = data2.drop(col);
			data3 = data3.drop(col);
			data4 = data4.drop(col);
		}
//		data1.show(5);
//		data2.show(6);
//		data3.show(7);
//		data4.show(8);

		System.out.println("Column drops completed");
//Join the dataset to create a single
		Dataset<Row> inputData =data1.join(data2,"songId");
		Dataset<Row> outputData =data3.join(data4,"notificationId");

		inputData.show(5);
		outputData.show(6);

		inputData = inputData.na().drop();
		inputData.toDF().write().mode(SaveMode.Overwrite).csv(tempPath+ Constants.TEMPINPUT_FOLDER);

		System.out.println("save training dataset completed");

		outputData.toDF().write().mode(SaveMode.Overwrite).csv(tempPath+ Constants.TEMPOUTPUT_FOLDER);

		System.out.println("save testing dataset completed");


		session.close();
	}

}

