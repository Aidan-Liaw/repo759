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
    results = list(csv.reader(open("./runtime.csv", "r")))
    results = numpy.transpose(results)
    
    sizes = 2 ** results[0].astype(int)
    runtimes = results[1].astype(float)

    figure = plt.figure(figsize=(11, 8.5))
    ax = plt.gca()

    ax.scatter(sizes, runtimes)
    ax.set_title("Inclusive Scan Performance")
    ax.set_xlabel("Input Size (unitless)")
    ax.set_xscale('function', functions=(log,power))
    ax.set_xticklabels([rf"$2^{{{j:.0f}}}$" for j in numpy.log2(sizes)])
    ax.set_xticks(sizes)
    ax.set_ylabel("Runtime (ms)")
    ax.set_yscale("log")

    # Fit power-law: y = a * x^b  (straight line on log-log axes)
    logx = numpy.log(sizes)
    logy = numpy.log(runtimes)
    b, loga = numpy.polyfit(logx, logy, 1)     # logy = b*logx + loga
    a = numpy.exp(loga)

    x_fit = numpy.geomspace(sizes.min(), sizes.max(), 200)
    y_fit = a * (x_fit ** b)

    ax.plot(x_fit, y_fit, linestyle='--', linewidth=1,
        label=rf"fit: $y \approx {a:.3g}\,x^{{{b:.3g}}}$")
    ax.legend()

    plt.savefig("task1.pdf", format="pdf")


