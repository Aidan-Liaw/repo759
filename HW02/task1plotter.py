import matplotlib.pyplot as plt
import numpy
import csv
import pandas

# REF: https://stackoverflow.com/questions
# /24606650/reading-csv-file-and-inserting-it-into-2d-list-in-python

# REF: https://www.geeksforgeeks.org
# /python/matplotlib-pyplot-scatter-in-python/

# REF: https://www.geeksforgeeks.org
# /data-visualization/exporting-plots-to-pdf-matplotlib/

if __name__ == "__main__":
    results = list(csv.reader(open("./runtime.csv", "r")))
    numpy.transpose(results)

    plt.scatter(results[0], results[1])
    plt.title("Input Size (unitless) Against Runtime (ms)")
    plt.xlabel("Input Size (unitless")
    plt.ylabel("Runtime (ms)")
    plt.savefig("task1.pdf", format="pdf")


