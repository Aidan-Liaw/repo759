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
    results_512 = list(csv.reader(open("runtime_512.csv", "r")))[1:]
    print(results_512)
    results_512 = numpy.transpose(results_512)

    results_16 = list(csv.reader(open("runtime_16.csv", "r")))[1:]
    results_16 = numpy.transpose(results_16)

    sizes = 2 ** results_512[0].astype(int)
    runtimes_512 = results_512[1].astype(float)
    runtimes_16 = results_16[1].astype(float)

    figure = plt.figure(figsize=(11, 8.5))
    ax = plt.gca()

    ax.plot(sizes, runtimes_512, "bo", linestyle='-', label="512 Threads/Block")
    ax.plot(sizes, runtimes_16, "r+", linestyle='-', label="16 Threads/Block")
    ax.set_title("vscale Kernel Function Performance")
    ax.set_xlabel("Input Size (unitless)")
    ax.set_xscale('function', functions=(log,power))
    ax.set_xticklabels([rf"$2^{{{j:.0f}}}$" for j in numpy.log2(sizes)])
    ax.set_xticks(sizes)
    ax.set_ylabel("Runtime (ms)")

    ax.legend()

    plt.savefig("task3.pdf", format="pdf")


