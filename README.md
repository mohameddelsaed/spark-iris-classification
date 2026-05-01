# Apache Spark — Iris Flower Classification

**Course:** Selected Topics in Information Systems — Lecture 9 (Apache Spark)
**Faculty of Computers and Information, Mansoura University** — 4th year, second term

This is the practical task for **Lecture 9 (Apache Spark)**. The lecture asks us to:

> bring sample data, choose a classification algorithm and apply it to that data using Spark.

The deliverable is a single Jupyter notebook,
[`spark_iris_classification.ipynb`](spark_iris_classification.ipynb), that:

1. Creates a `SparkSession` using the **same boilerplate shown in the lecture** (slide
   *Simple PySpark app.*):
   ```python
   from pyspark.sql import SparkSession
   spark = (
       SparkSession.builder
       .master("local[1]")
       .appName("SparkByExamples.com")
       .getOrCreate()
   )
   rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
   print("RDD count =", rdd.count())   # 5
   ```
2. Loads the **Iris** sample dataset (150 rows, 4 numeric features, 3 classes).
3. Builds a Spark MLlib pipeline:
   `VectorAssembler` (4 features → one `features` vector) +
   `StringIndexer` (`species` → numeric `label`).
4. Splits the data into train / test (80 / 20, `seed=42`).
5. Trains two **classification algorithms** from `pyspark.ml.classification`:
   - `LogisticRegression` (multinomial)
   - `RandomForestClassifier` (50 trees, max depth 5)
6. Evaluates each model with `MulticlassClassificationEvaluator`
   (`accuracy` and `f1`) and reports feature importances for the random forest.

## Results

| Model               | Accuracy | F1   |
| ------------------- | -------- | ---- |
| Logistic Regression | 1.0000   | 1.00 |
| Random Forest       | 1.0000   | 1.00 |

(Test set = 24 held-out rows. Iris is a small, well-separated dataset, so reaching
100 % on the test split is expected.)

Random-forest feature importances (top first):

```
petal_width    0.4718
petal_length   0.4057
sepal_length   0.0986
sepal_width    0.0238
```

## How to run

1. Install Java 8/11/17 and Python 3.9+.
2. Install the Python dependencies:
   ```bash
   pip install pyspark==3.5.3 jupyter nbformat
   ```
3. Open the notebook and run all cells:
   ```bash
   jupyter notebook spark_iris_classification.ipynb
   ```
   Or execute it from the command line:
   ```bash
   jupyter nbconvert --to notebook --execute spark_iris_classification.ipynb \
       --output spark_iris_classification.ipynb
   ```

The notebook downloads the Iris CSV from the UCI repository on first run and caches
it under `data/iris.csv`.

## Files

| Path                                  | Description                                  |
| ------------------------------------- | -------------------------------------------- |
| `spark_iris_classification.ipynb`     | The executed notebook (with outputs).        |
| `spark_iris_classification.html`      | HTML export of the executed notebook.        |
| `build_notebook.py`                   | Script that generated the notebook cells.   |
| `data/iris.csv`                       | Iris dataset (downloaded on first run).      |
| `requirements.txt`                    | Python dependencies.                          |
