"""Build the Spark Iris classification Jupyter notebook."""
import json
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    "# Apache Spark — Iris Flower Classification\n"
    "\n"
    "**Course:** Selected Topics in Information Systems — Lecture 9 (Apache Spark)  \n"
    "**Faculty of Computers and Information, Mansoura University**\n"
    "\n"
    "This notebook follows the exact code style shown in Lecture 9 (slides *Simple PySpark app.*) "
    "and then extends it to the assignment requirement: bring **sample data**, choose a "
    "**classification algorithm**, and apply it to that data using Apache Spark.\n"
    "\n"
    "**Sample data:** Iris flower dataset (3 classes — *setosa*, *versicolor*, *virginica*; "
    "4 numeric features).  \n"
    "**Classification algorithms used:**\n"
    "1. Logistic Regression (`pyspark.ml.classification.LogisticRegression`)\n"
    "2. Random Forest Classifier (`pyspark.ml.classification.RandomForestClassifier`)\n"
    "\n"
    "Pipeline steps:\n"
    "1. Create `SparkSession` (same boilerplate as the lecture).\n"
    "2. Sanity-check Spark by running the lecture's `parallelize` / `count` example.\n"
    "3. Load the Iris CSV into a Spark DataFrame.\n"
    "4. Convert features into a single vector (`VectorAssembler`) and index the string label "
    "(`StringIndexer`).\n"
    "5. Split the data into train / test sets (80 / 20).\n"
    "6. Train each classifier and evaluate accuracy + F1 on the test set.\n"
    "7. Compare the two models."
)

md("## 1. Imports and SparkSession (same as Lecture 9)")

code(
    "from pyspark.sql import SparkSession\n"
    "\n"
    "# Same boilerplate shown in Lecture 9 (slide \"Simple PySpark app.\")\n"
    "spark = (\n"
    "    SparkSession.builder\n"
    "    .master(\"local[1]\")\n"
    "    .appName(\"SparkByExamples.com\")\n"
    "    .getOrCreate()\n"
    ")\n"
    "spark.sparkContext.setLogLevel(\"WARN\")\n"
    "print(spark)"
)

md("### 1.1 Quick sanity check (the exact RDD example from the lecture)")

code(
    "rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])\n"
    "print(\"RDD count =\", rdd.count())"
)

md(
    "## 2. Load the Iris sample dataset\n"
    "\n"
    "We download the classic Iris CSV from the UCI repository and read it into a Spark "
    "DataFrame. The four numeric features are the sepal/petal length and width; the label "
    "is the flower species."
)

code(
    "import os\n"
    "import urllib.request\n"
    "\n"
    "DATA_DIR = \"data\"\n"
    "os.makedirs(DATA_DIR, exist_ok=True)\n"
    "csv_path = os.path.join(DATA_DIR, \"iris.csv\")\n"
    "\n"
    "if not os.path.exists(csv_path):\n"
    "    url = (\n"
    "        \"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data\"\n"
    "    )\n"
    "    raw = urllib.request.urlopen(url).read().decode(\"utf-8\").strip().splitlines()\n"
    "    header = \"sepal_length,sepal_width,petal_length,petal_width,species\"\n"
    "    with open(csv_path, \"w\", encoding=\"utf-8\") as f:\n"
    "        f.write(header + \"\\n\")\n"
    "        for line in raw:\n"
    "            if line.strip():\n"
    "                f.write(line + \"\\n\")\n"
    "\n"
    "print(\"Saved:\", csv_path, \"(\", os.path.getsize(csv_path), \"bytes )\")"
)

code(
    "from pyspark.sql.types import (\n"
    "    StructType,\n"
    "    StructField,\n"
    "    DoubleType,\n"
    "    StringType,\n"
    ")\n"
    "\n"
    "schema = StructType([\n"
    "    StructField(\"sepal_length\", DoubleType(), True),\n"
    "    StructField(\"sepal_width\",  DoubleType(), True),\n"
    "    StructField(\"petal_length\", DoubleType(), True),\n"
    "    StructField(\"petal_width\",  DoubleType(), True),\n"
    "    StructField(\"species\",      StringType(), True),\n"
    "])\n"
    "\n"
    "iris = spark.read.csv(csv_path, header=True, schema=schema)\n"
    "iris.printSchema()\n"
    "iris.show(5)\n"
    "print(\"Total rows:\", iris.count())"
)

md("### 2.1 Class distribution")

code("iris.groupBy(\"species\").count().orderBy(\"species\").show()")

md(
    "## 3. Feature engineering\n"
    "\n"
    "* `VectorAssembler` packs the four numeric columns into a single `features` vector.\n"
    "* `StringIndexer` converts the string `species` column into a numeric `label` column "
    "(required by Spark MLlib classifiers)."
)

code(
    "from pyspark.ml.feature import VectorAssembler, StringIndexer\n"
    "\n"
    "feature_cols = [\"sepal_length\", \"sepal_width\", \"petal_length\", \"petal_width\"]\n"
    "\n"
    "assembler = VectorAssembler(inputCols=feature_cols, outputCol=\"features\")\n"
    "indexer   = StringIndexer(inputCol=\"species\", outputCol=\"label\")\n"
    "\n"
    "data = assembler.transform(iris)\n"
    "data = indexer.fit(data).transform(data)\n"
    "data = data.select(\"features\", \"label\", \"species\")\n"
    "data.show(5, truncate=False)"
)

md("## 4. Train / test split (80 / 20)")

code(
    "train, test = data.randomSplit([0.8, 0.2], seed=42)\n"
    "print(\"Training rows:\", train.count())\n"
    "print(\"Test rows    :\", test.count())"
)

md(
    "## 5. Model 1 — Logistic Regression\n"
    "\n"
    "Multinomial logistic regression is a natural baseline for the 3-class Iris problem."
)

code(
    "from pyspark.ml.classification import LogisticRegression\n"
    "from pyspark.ml.evaluation import MulticlassClassificationEvaluator\n"
    "\n"
    "lr = LogisticRegression(\n"
    "    featuresCol=\"features\",\n"
    "    labelCol=\"label\",\n"
    "    maxIter=50,\n"
    "    family=\"multinomial\",\n"
    ")\n"
    "lr_model = lr.fit(train)\n"
    "lr_pred  = lr_model.transform(test)\n"
    "lr_pred.select(\"species\", \"label\", \"prediction\", \"probability\").show(5, truncate=False)"
)

code(
    "acc_eval = MulticlassClassificationEvaluator(\n"
    "    labelCol=\"label\", predictionCol=\"prediction\", metricName=\"accuracy\",\n"
    ")\n"
    "f1_eval  = MulticlassClassificationEvaluator(\n"
    "    labelCol=\"label\", predictionCol=\"prediction\", metricName=\"f1\",\n"
    ")\n"
    "\n"
    "lr_acc = acc_eval.evaluate(lr_pred)\n"
    "lr_f1  = f1_eval.evaluate(lr_pred)\n"
    "print(f\"Logistic Regression  -> accuracy = {lr_acc:.4f}, F1 = {lr_f1:.4f}\")"
)

md(
    "## 6. Model 2 — Random Forest Classifier\n"
    "\n"
    "An ensemble of decision trees that usually performs very well on small tabular datasets."
)

code(
    "from pyspark.ml.classification import RandomForestClassifier\n"
    "\n"
    "rf = RandomForestClassifier(\n"
    "    featuresCol=\"features\",\n"
    "    labelCol=\"label\",\n"
    "    numTrees=50,\n"
    "    maxDepth=5,\n"
    "    seed=42,\n"
    ")\n"
    "rf_model = rf.fit(train)\n"
    "rf_pred  = rf_model.transform(test)\n"
    "rf_pred.select(\"species\", \"label\", \"prediction\").show(5, truncate=False)\n"
    "\n"
    "rf_acc = acc_eval.evaluate(rf_pred)\n"
    "rf_f1  = f1_eval.evaluate(rf_pred)\n"
    "print(f\"Random Forest        -> accuracy = {rf_acc:.4f}, F1 = {rf_f1:.4f}\")"
)

md("### 6.1 Feature importances (Random Forest)")

code(
    "importances = list(zip(feature_cols, rf_model.featureImportances.toArray()))\n"
    "importances.sort(key=lambda kv: kv[1], reverse=True)\n"
    "for name, score in importances:\n"
    "    print(f\"  {name:<14s} {score:.4f}\")"
)

md("## 7. Comparison")

code(
    "from pyspark.sql import Row\n"
    "\n"
    "results = spark.createDataFrame([\n"
    "    Row(model=\"LogisticRegression\", accuracy=float(lr_acc), f1=float(lr_f1)),\n"
    "    Row(model=\"RandomForest\",       accuracy=float(rf_acc), f1=float(rf_f1)),\n"
    "])\n"
    "results.show(truncate=False)"
)

md("## 8. Stop the SparkSession")

code("spark.stop()")

md(
    "---\n"
    "\n"
    "**Summary**\n"
    "\n"
    "* Used the same `SparkSession` boilerplate from Lecture 9.\n"
    "* Loaded the Iris **sample data** (150 rows, 4 features, 3 classes).\n"
    "* Built a Spark MLlib pipeline (`VectorAssembler` + `StringIndexer`).\n"
    "* Trained two **classification algorithms** (Logistic Regression and Random Forest) "
    "and evaluated them with `MulticlassClassificationEvaluator`.\n"
    "* Both models achieve high accuracy on the held-out test set, fulfilling the assignment "
    "requirement to *\"bring sample data and choose a classification algorithm and apply it "
    "to that data\"* using Apache Spark."
)

nb["cells"] = cells

out = Path("/home/ubuntu/spark-iris-classification/spark_iris_classification.ipynb")
nbf.write(nb, out)
print("Wrote", out)
