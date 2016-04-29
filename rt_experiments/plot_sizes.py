import argparse
from common import read_table_lengths
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    with open(args.filename, 'rb') as fp:
        lens = read_table_lengths(fp)

    plt.hist(list(lens.values()))
    plt.show()
