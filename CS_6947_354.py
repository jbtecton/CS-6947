from tecton import spark_batch_config, BatchSource
from datetime import datetime, timedelta
from tecton import Attribute, Entity, batch_feature_view, Aggregate, FeatureService
from tecton.types import Field, String, Int64
from tecton.aggregation_functions import first_distinct, approx_count_distinct
from pyspark.sql.functions import to_timestamp, lit

@spark_batch_config(supports_time_filtering=True)
def dsf(spark, filter_context):
    from pyspark.sql import functions as F
    
    # Calculate relative dates based on today
    today = datetime.now()
    date_4_days_ago = (today - timedelta(days=4)).replace(hour=0, minute=0, second=0, microsecond=0)
    date_3_days_ago = (today - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    date_2_days_ago = (today - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    date_1_day_ago = (today - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
    
    cols = ["user_id", "timestamp", "value"]
    data = [
        ["user_1", date_2_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 1],
        ["user_1", date_2_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 1],
        ["user_1", (date_2_days_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 2],
        ["user_1", (date_2_days_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 3],
        ["user_1", (date_1_day_ago).strftime("%Y-%m-%d %H:%M:%S"), 4],
        ["user_1", (date_1_day_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 5],
        ["user_1", (date_1_day_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 6],
        ["user_1", today_start.strftime("%Y-%m-%d %H:%M:%S"), 7],
        ["user_1", (today_start + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 8],
        ["user_1", (today_start + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 9],
        ["user_1", date_4_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 10],
        ["user_1", date_4_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 10],
        ["user_1", (date_4_days_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 11],
        ["user_1", (date_4_days_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 12],
        ["user_1", date_3_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 13],
        ["user_1", (date_3_days_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 14],
        ["user_1", (date_3_days_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 15],
        ["user_1", date_2_days_ago.strftime("%Y-%m-%d %H:%M:%S"), 16],
        ["user_1", (date_2_days_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 17],
        ["user_1", (date_2_days_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 18],
        ["user_1", date_1_day_ago.strftime("%Y-%m-%d %H:%M:%S"), 19],
        ["user_1", (date_1_day_ago + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"), 20],
        ["user_1", (date_1_day_ago + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%d %H:%M:%S"), 21],
        ["user_1", today_start.strftime("%Y-%m-%d %H:%M:%S"), 22], 
        ["user_2", (today_start + timedelta(hours=13)).strftime("%Y-%m-%d %H:%M:%S"), None],
        ["user_2", (date_2_days_ago + timedelta(hours=14)).strftime("%Y-%m-%d %H:%M:%S"), None],
        ["user_2", (date_2_days_ago + timedelta(hours=15)).strftime("%Y-%m-%d %H:%M:%S"), None],
        ["user_2", (date_2_days_ago + timedelta(hours=16)).strftime("%Y-%m-%d %H:%M:%S"), None],
    ]
    df = spark.createDataFrame(data, cols)
    df = df.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))
    return df


ds = BatchSource(name="CS_6947_354", batch_config=dsf)
user_entity = Entity(name="user_CS_6947_354", join_keys=[Field("user_id", String)])

@batch_feature_view(
    mode="pyspark",
    sources=[ds],
    entities=[user_entity],
    aggregation_interval=timedelta(days=7),
    timestamp_field="timestamp",
    tags={
        "domain": "prod",
        "team": "data-science", 
        "environment": "test",
        "test_tag": "test_value",
        "updated_tag": "updated_value",
        "third_new_tag": "third_new_value"
    },
    offline=True,
    online=True,
    feature_start_time=datetime.now() - timedelta(days=6),
    features=[
        Aggregate(
            input_column=Field("value", Int64),
            function="count",
            time_window=timedelta(days=7),
            name="value_count_7d",
        ),
        Aggregate(
            input_column=Field("value", Int64),
            function=approx_count_distinct(10),  # ← Added count_distinct aggregation
            time_window=timedelta(days=7),
            name="value_count_distinct_7d",
        ),
        Aggregate(
            input_column=Field("value", Int64),
            function=first_distinct(2),
            time_window=timedelta(days=7),
            name="value_first_3_distinct_7d",
        ),
        Aggregate(
            input_column=Field("value", Int64),
            function="last",
            time_window=timedelta(days=7),
            name="value_last_7d",
        ),
    ],
)
def feature_view_a_CS_6947_354(input_table):
    return input_table[["user_id", "timestamp", "value"]]

feature_service_CS_6947_354 = FeatureService(
    name="feature_service_CS_6947_354",
    description="Testing NaN versus zero output in aggregations from offline versus online store",
    online_serving_enabled=True,
    features=[
        feature_view_a_CS_6947_354
    ],
)


@batch_feature_view(
    name="feature_view_b_CS_6947_354",
    mode="pyspark",
    sources=[ds],
    entities=[user_entity],
    feature_start_time=datetime.now() - timedelta(days=7),
    batch_schedule=timedelta(days=2),
    online=True,
    offline=True,
    #aggregation_interval=timedelta(days=1),
    timestamp_field="timestamp",
    features=[
        Attribute("simple_value", Int64)
    ]
)
def feature_view_b_CS_6947_354(df):
    return df.select("user_id", "timestamp", "value").withColumnRenamed("value", "simple_value")
