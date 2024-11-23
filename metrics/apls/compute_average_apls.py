import os
import sys
import numpy as np


def compute_average_apls(txt_path):
    """ Compute average apls using apls for single image """
    count = 0
    sum = 0
    for txt_name in os.listdir(txt_path):
        with open(os.path.join(txt_path, txt_name), "r") as f:
            outcome = f.readline().strip("\n").split(" ")
            average_apls = float(outcome[-1])
            if not np.isnan(average_apls):
                count += 1
                sum += average_apls

    apls = sum / count
    return apls


if __name__ == "__main__":
    txt_path = sys.argv[1]
    metric_save_path = sys.argv[2]
    apls = compute_average_apls(txt_path=txt_path)
    print(f"apls: {apls}\n")
    with open(os.path.join(metric_save_path, "result.txt"), 'a') as f:
        f.write(f"apls: {apls}\n")
