import os
import sys
import numpy as np


def compute_average_topo(txt_path):
    """ Compute average topo using topo for single image """
    count = 0
    sum_P = 0
    sum_R = 0
    for txt_name in os.listdir(txt_path):
        with open(os.path.join(txt_path, txt_name), "r") as f:
            try:
                precisions_str, recall_str = f.readline().strip("\n").split(" ")
                precision = precisions_str.strip('precision=')
                recall = recall_str.strip('recall=')
            except:
                precisions, recall = 0, 0
            
            count += 1
            sum_P += float(precision)
            sum_R += float(recall)

    overall_P = sum_P / count
    overall_R = sum_R / count
    overall_F1 = 2 * overall_P * overall_R / (overall_R + overall_P)
    return f'precision: {overall_P} recall: {overall_R} F1: {overall_F1}'


if __name__ == "__main__":
    txt_path = sys.argv[1]
    metric_save_path = sys.argv[2]
    topo = compute_average_topo(txt_path=txt_path)
    print(f"topo: {topo}\n")
    with open(os.path.join(metric_save_path, "result.txt"), 'a') as f:
        f.write(f"topo:\n{topo}\n")
