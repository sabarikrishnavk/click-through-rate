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

java -jar target/click-through-rate-jar-with-dependencies.jar input output
