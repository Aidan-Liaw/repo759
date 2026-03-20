import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
import csv

import numpy as np
import pandas

# REF: https://stackoverflow.com/questions
# /24606650/reading-csv-file-and-inserting-it-into-2d-list-in-python

# REF: https://www.geeksforgeeks.org
# /python/matplotlib-pyplot-scatter-in-python/

# REF: https://www.geeksforgeeks.org
# /data-visualization/exporting-plots-to-pdf-matplotlib/

# REF: https://stackoverflow.com/questions
# /18773662/scatter-plot-logarithmic-scale

# REF: https://stackoverflow.com/questions
# /65319997/how-to-set-the-tick-scale-as-the-power-of-2-in-matplotlib

# ACKNOWLEDGEMENT: The problem was solved by me 80% and ChatGPT 20%
# The AI model was asked to fix an issue with the x-axis ticks not formatting
# correctly, and to implement a line of best fit (based on the plot being an
# exponent-log graph, and to fix the page size output to US Letter landscape

def log(x):
    return numpy.log2(x)

def power(x):
    return 2**x

if __name__ == "__main__":
    results = list(csv.reader(open("./runtime_1.csv", "r")))[1:]
    print(results)
    results = numpy.transpose(results)

    thread_count = results[0].astype(int)
    runtimes = results[1].astype(float)

    figure = plt.figure(figsize=(11, 8.5))
    ax = plt.gca()

    ax.plot(thread_count, runtimes, "bo", linestyle='-', label="Matrix Multiplication Runtime With 1024 Sized Input")
    ax.grid(which="both")
    ax.set_title("Task 1: Parallel Matrix Multiplication Runtime")
    ax.set_xticks(thread_count)
    ax.set_xlabel("Thread count (unitless)")
    ax.set_ylabel("Runtime (ms)")

    ax.legend()

    plt.savefig("hw8_task1.pdf", format="pdf")


