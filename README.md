# K-Means Clustering Algorithm Performance Comparison

## Overview
This project evaluates the performance, scalability, and efficiency of three implementations of the K-Means clustering algorithm: Pure Python, Numpy, and Cython. Through detailed analysis and profiling across various dataset sizes and dimensions, we aim to identify optimal strategies for efficient data clustering, particularly in high-dimensional spaces.

## Objective
The goal is to determine which implementation of the K-Means algorithm is the most efficient in terms of execution speed and memory usage, and how well these implementations scale with increasing data complexity.

## Implementation Details
- **Pure Python Version**: A straightforward implementation using standard Python libraries, emphasizing simplicity and readability.
- **Numpy Version**: Optimizes computation through Numpy's array operations, with further enhancements from Numexpr for improved speed and reduced memory overhead.
- **Cython Version**: Compiles Python code into C using Cython, targeting maximum computational efficiency and speed, especially for large datasets.

## Experimental Setup
- **Dataset Parameters**: Datasets of 100, 1,000, and 10,000 points in dimensions 2, 5, and 10 were used to test scalability and performance.
- **Clusters**: The number of clusters was consistently set to 3 for all tests.
- **Data Generation**: Uniform random data generation ensured controlled and fair comparisons.
- **Environment**: A consistent hardware and software environment was maintained for all tests, with specifications documented for reproducibility.

## Profiling and Analysis
- **Execution Time**: Measured with `%timeit` in Jupyter notebooks, averaging results over multiple runs.
- **Memory Usage**: Assessed using `memory_profiler` to note differences in resource management.
- **Scalability**: Evaluated based on changes in execution time and memory usage with increasing dataset sizes and dimensions.

## Results and Conclusions
- **Performance**: Cython showed the highest speed, followed by Numpy. The Pure Python version lagged significantly, particularly with larger and more complex datasets.
- **Memory Efficiency**: Numpy and Cython were more memory-efficient than Pure Python, with Numpy's performance further enhanced by Numexpr.
- **Scalability**: Numpy and Cython demonstrated superior scalability, maintaining efficient execution times and memory usage across varied test conditions.
- **Recommendations**: The Numpy version is recommended for small to medium datasets for its balance of performance and usability. For large datasets or performance-critical applications, Cython is the preferred choice.

