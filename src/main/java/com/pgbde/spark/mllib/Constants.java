package com.pgbde.spark.mllib;

public class Constants {
    public static final String LOCAL ="local" ;
    public static final String DEF_ACCESSKEY = "accesskey";
    public static final String DEF_SECRETKEY = "secretkey";
    public static final String SAVEMODE_I = "1";
    public static final int PROGRAM_CHECKPOINT_1 = 1; //Data prep
    public static final int PROGRAM_CHECKPOINT_2 = 2; //Training data and Predictions using training RDD

    public static final String ACTIVITY_FOLDER = "/activity/";
    public static final String METADATA_FOLDER = "/newmetadata/";
    public static final String NOTIFICATIONCLICK_FOLDER = "/notification_clicks/";
    public static final String NOTIFICATIONACTOR_FOLDER = "/notification_actor/";

    public static final String AWS = "aws";
    public static final String TEMPINPUT_FOLDER = "/input/";
    public static final String TEMPOUTPUT_FOLDER = "/output/" ;
    public static final String MODELOUTPUT_FOLDER = "/model/" ;


    public static final String CTR_FOLDER = "/ctr/" ;
}
