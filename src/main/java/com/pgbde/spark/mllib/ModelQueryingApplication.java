package com.pgbde.spark.mllib;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.SparkSession;

public class ModelQueryingApplication {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        String modelPath =args[0] ;// "output";
        int artistId =   (args.length >= 1) ?Integer.parseInt(args[1]) : 535722;
        String userIdStr = "5525c71b6213340569f3aa1abc225514";//34
        int topArtistCount =   (args.length >= 2) ?Integer.parseInt(args[2]) : 5;


        SparkSession session = SparkSession.builder().master("local")
                    .config("fs.s3a.connection.ssl.enabled", "false")
                    .config("spark.network.timeout", "600s")
                    .config("spark.executor.heartbeatInterval", "500s")
                    .getOrCreate();


        JavaSparkContext jsc = new JavaSparkContext(session.sparkContext());

        MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),modelPath);

        Rating[] results = sameModel.recommendUsers(artistId, topArtistCount);

        for(Rating result: results){
            System.out.println(result);
        }
        jsc.close();
    }
}
