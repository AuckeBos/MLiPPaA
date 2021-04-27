# MLiPPaA

This repo contains the code for the exam project of the course Machine Learning in Particle Physics &amp; Astronomy. The
code is structured in a format that is required by the assignment, it is explained below.
The [assignment](Assignment.pdf) itself can be found in the root of this repo.

# Installation

* Install poetry: `sudo apt-get install poetry`
* Run `poetry install` to install the virtual environment
* Run `poetry python exam.py` to run the project. Choose which sub-assignment to run by (un)commenting the relevant
  lines at the bottom of the file
* \[Optional\] Run `poetry shell` to enter the virtual environment

# Structure

The assignment [pdf](Assignment.pdf) requires a Handin consisting of 4 elements:

1. The first element is the paper that describes the problem, algorithm, design choices, encountered problems, and an
   evaluation performance. It can be found in [Element 1](Element1) ([pdf](Element1/paper.pdf)).
2. [Element 2](Element2) contains the sourcecode. Its entrypoint is [main.py](Element2/main.py). A small description of
   the classes:

    - The [BaseClassifier](Element2/BaseClassification.py) is a base classifier used by all implementations. It defines
      functions for losses, metrics, data loading, model compilation, testing and training. Specific implementations
      extend and override its functionalities.
    - The [BinaryClassifier](Element2/BinaryClassification.py) defines a binary classifier for the problem.
    - The [MultiClassifier](Element2/MultiClassification.py) defines the basic multiclass classifier for the problem.
    - The [RecurrentClassifier](Element2/RecurrentClassification.py) defines a more sophisticated multiclass classifier
      for the problem.
    - The [Evaluator](Element2/Evaluator.py) provides functionality to evaluate different models and configurations.
      They are extensively discussed in the paper.

   As said, *main.py* defines the entrypoint. The assignment conists of 4 subtasks. *main.py* defines 4 functions, 1 for
   each task:

    - *assignment_a()* Evaluates a simple binary classifier.
    - *assignment_b()* Evaluates a basic multiclass classifier.
    - *assignment_c()* Compares the performance of the classifiers of *a()* and *b()*.
    - *assignment_d()* Evaluates a more complex recurrent multiclass classifier.

   For these tasks, we have chosen the configurations that worked well, according to Tables I-III in
   the [paper](Element1/paper.pdf).
3. [Element 3](Element3) provides a script to read a previously trained model, and test its performance on a test set.
   It consumes several configuration variables, and computes a csv file with classification probabilities.
4. [Element 4](Element4) provides the predictions on the testset as provided in the course. These predictions are
   created using the [script](Element3/read_and_run.py) of Element 3.\
   **Note that the directory is empty, to reduce the size of the repo. The csvs (binary and multiclass) are handed in at
   the assignment of the course**

# Tables

For completeness, Tables I - III of the paper are provided below.

TABLE I. Performance comparison for the *MultiClassifier* for all combinations of 4 design choices (see Section IV A of
the [paper](Element1/paper.pdf)).

| Configuration index | 0         | 1         | 2         | 3         | 4         | 5         | 6         | 7         | 8         | 9         | 10        | 11        | 12        | 13        | 14        | 15        |           |
   |---------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Apply Bayes         | T         | T         | T         | T         | T         | T         | T         | T         | F         | F         | F         | F         | F         | F         | F         | F         |           |
| Rebalance test      | T         | T         | T         | T         | F         | F         | F         | F         | T         | T         | T         | T         | F         | F         | F         | F         |           |
| Rebalance train/val | T         | T         | F         | F         | T         | T         | F         | F         | T         | T         | F         | F         | T         | T         | F         | F         |           |
| Weighted loss       | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | AVG       |
| Validation loss     | 1.620     | 1.309     | 1.448     | 0.944     | 1.621     | 1.301     | 1.449     | 0.946     | 1.607     | 1.305     | 1.448     | 0.949     | 1.599     | 1.301     | 1.449     | 0.944     | 1.327     |
| Test loss           | 1.317     | 1.367     | 1.428     | 1.295     | 0.979     | 1.416     | 1.117     | 0.942     | 1.410     | 1.292     | 1.900     | 1.559     | 0.984     | 1.397     | 1.121     | 0.940     | 1.279     |
| Validation accuracy | 0.345     | 0.407     | 0.582     | 0.624     | 0.351     | 0.412     | 0.590     | 0.623     | 0.351     | 0.412     | 0.584     | 0.620     | 0.349     | 0.411     | 0.586     | 0.624     | 0.492     |
| Test accuracy       | 0.400     | 0.407     | 0.356     | 0.410     | 0.615     | 0.417     | 0.595     | 0.627     | 0.356     | 0.416     | 0.213     | 0.317     | 0.614     | 0.434     | 0.591     | 0.628     | 0.462     |
| Validation f1       | 0.139     | 0.184     | 0.591     | 0.608     | 0.142     | 0.196     | 0.594     | 0.609     | 0.150     | 0.187     | 0.590     | 0.604     | 0.142     | 0.199     | 0.596     | 0.608     | 0.384     |
| Test f1             | 0.361     | 0.326     | 0.335     | 0.405     | 0.434     | 0.390     | 0.316     | 0.411     | 0.330     | 0.409     | 0.156     | 0.267     | 0.432     | 0.391     | 0.326     | 0.413     | 0.356     |
| Epochs              | 46        | 77        | 62        | 61        | 37        | 60        | 44        | 58        | 39        | 69        | 56        | 68        | 48        | 62        | 47        | 69        | 56.4375   |
| LR                  | 4.096E-07 | 1.678E-09 | 1.049E-08 | 4.096E-07 | 2.560E-06 | 1.024E-06 | 4.096E-07 | 4.096E-07 | 2.560E-06 | 2.621E-08 | 4.096E-07 | 1.678E-09 | 4.096E-07 | 1.638E-07 | 1.638E-07 | 4.096E-07 | 5.862E-07 |

TABLE II. Performance comparison for the *RecurrentClassifier* for all combinations of 4 design choices (see Section IV
A of the [paper](Element1/paper.pdf).

| Configuration index | 0         | 1         | 2         | 3         | 4         | 5         | 6         | 7         | 8         | 9         | 10        | 11        | 12        | 13        | 14        | 15        |           |
|---------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Apply Bayes         | T         | T         | T         | T         | T         | T         | T         | T         | F         | F         | F         | F         | F         | F         | F         | F         |           |
| Rebalance test      | T         | T         | T         | T         | F         | F         | F         | F         | T         | T         | T         | T         | F         | F         | F         | F         |           |
| Rebalance train/val | T         | T         | F         | F         | T         | T         | F         | F         | T         | T         | F         | F         | T         | T         | F         | F         |           |
| Weighted loss       | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | T         | F         | AVG       |
| Validation loss     | 1.755     | 1.400     | 1.672     | 1.056     | 1.780     | 1.378     | 1.683     | 1.053     | 1.743     | 1.397     | 1.691     | 1.050     | 1.832     | 1.400     | 1.676     | 1.058     | 1.476     |
| Test loss           | 1.384     | 1.464     | 1.503     | 1.374     | 1.099     | 1.589     | 1.370     | 1.051     | 1.516     | 1.393     | 2.574     | 1.666     | 1.090     | 1.604     | 1.383     | 1.057     | 1.445     |
| Validation accuracy | 0.301     | 0.340     | 0.549     | 0.581     | 0.273     | 0.356     | 0.545     | 0.582     | 0.287     | 0.341     | 0.545     | 0.584     | 0.240     | 0.341     | 0.545     | 0.581     | 0.437     |
| Test accuracy       | 0.350     | 0.344     | 0.304     | 0.359     | 0.570     | 0.343     | 0.551     | 0.586     | 0.296     | 0.344     | 0.140     | 0.252     | 0.574     | 0.374     | 0.550     | 0.584     | 0.408     |
| Validation f1       | 0.060     | 0.012     | 0.554     | 0.568     | 0.034     | 0.049     | 0.551     | 0.567     | 0.062     | 0.012     | 0.551     | 0.567     | 0.036     | 0.010     | 0.549     | 0.566     | 0.297     |
| Test f1             | 0.298     | 0.242     | 0.272     | 0.334     | 0.354     | 0.327     | 0.233     | 0.336     | 0.268     | 0.327     | 0.109     | 0.211     | 0.338     | 0.309     | 0.232     | 0.332     | 0.283     |
| Epochs              | 52        | 31        | 42        | 30        | 43        | 74        | 41        | 50        | 62        | 40        | 32        | 48        | 31        | 39        | 38        | 47        | 43.75     |
| LR                  | 6.400E-06 | 2.560E-06 | 2.560E-06 | 6.400E-06 | 1.024E-06 | 6.400E-06 | 6.400E-06 | 1.638E-07 | 1.024E-06 | 1.024E-06 | 6.400E-06 | 1.024E-06 | 1.024E-06 | 1.024E-06 | 6.400E-06 | 1.024E-06 | 3.178E-06 |

TABLE III. Performance comparison on binary classification between our *BinaryClassifier* and our *MultiClassifier* for
the combinations of 2 design choices (see Section IV A of the [paper](Element1/paper.pdf)). The rebalanced test set is
rebalanced to [96%, 4%]. The *Multiclassifier* scores better.

| Configuration index      | 3     | 5     | 6     | 12    |       |
|--------------------------|-------|-------|-------|-------|-------|
| Rebalance train/val      | F     | T     | F     | T     |       |
| Weighted loss            | F     | F     | T     | T     | AVG   |
| Binary F1 rebalanced     | 0.612 | 0.673 | 0.039 | 0.204 | 0.382 |
| Multi F1 rebalanced      | 0.534 | 0.713 | 0.346 | 0.599 | 0.548 |
| Binary F1 not rebalanced | 0.865 | 0.576 | 0.334 | 0.512 | 0.572 |
| Multi F1 not rebalanced  | 0.839 | 0.656 | 0.668 | 0.856 | 0.755 |