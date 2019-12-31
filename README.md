# click-through-rate

Create following folders inside the inputPath to keep the input files required for the program

newmetadata
notification_clicks
activity
notification_actor


Create CSV  files.
-----  

// Load the clickstreamdata-- four attributes - “User ID”, “Song ID”,  “Date” and “Timestamp.”
//5525c71b6213340569f3aa1abc225514,1533115844,_JUdlvPU,20180801
		String path1 = inputPath+ "activity";

//metadata folder -- “Song ID” to an “Artist ID.”A specific "Song ID" can be related to multiple "Artist IDs".
//Zil3cVnY,516437
		String path2 =inputPath+"newmetadata";

//Notification Clicks -Attributes: "Notification ID", "User ID", and "Date"
//9681,69f1004d6c2395cf556c76498e041d5e,20180826
		String path3 =inputPath+"notification_clicks";
		
//Notification Artists-- "Notification ID" and "Artist ID"
//9553,535722
		String path4 =inputPath+"notification_actor";
	
Build the package
----
mvn clean install
	
Run the program
----

Generated model is saved under "output" folder.

Refer AWS IAM to generate an aws credentials (access/secret keys)

java -jar target/click-through-rate-jar-with-dependencies.jar input/ output/ aws aws_accesskey aws_secretkey


nohup spark2-submit --class com.pgbde.spark.mllib.DataPreparationCTR --master yarn --deploy-mode client --executor-memory 2G --driver-memory 4G click-through-rate-jar-with-dependencies.jar s3a://bigdataanalyticsupgrad/ output/temp/ aws aws_accesskey aws_secretkey >>log.txt &
nohup spark2-submit --class com.pgbde.spark.mllib.DataAnalysisCTR --master yarn --deploy-mode client --executor-memory 2G --driver-memory 4G click-through-rate-jar-with-dependencies.jar s3a://bigdataanalyticsupgrad/ output/ctr/ aws aws_accesskey aws_secretkey >>log.txt &


Reference:

https://dataplatform.cloud.ibm.com/exchange/public/entry/view/99b857815e69353c04d95daefb3b91fa
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3175648861028866/48824497172554/657465297935335/latest.html
