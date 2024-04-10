import os
import numpy as np
import json
import argparse

def main():
    metrics = []

    parser = argparse.ArgumentParser(description="Gets average metrics from a folder of predict_results.json files")
    parser.add_argument("dir_path", help="The input file to process")
    args = parser.parse_args()

    # Print the value of the dir_path argument
    print(f"dir_path: {args.dir_path}")

    for root, dirs, files in os.walk(args.dir_path):
        for file in files: 
            if file == 'predict_results.json':
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                metrics.append([os.path.join(root, file), data['predict_overall_f1']])
    avg_metrics = np.mean([metric[1] for metric in metrics])
    stdev_metrics = np.std([metric[1] for metric in metrics])
    filename = os.path.join(args.dir_path, 'average_f1.txt')
    with open(filename, 'w', encoding='utf8') as f:
        for metric in metrics:
            f.write(' '.join([str(el) for el in metric]) + '\n')
        f.write(f"avg: {avg_metrics}")
        f.write(f"avg: {stdev_metrics}")
    suffix = f'_{round(avg_metrics, 2)}_{round(stdev_metrics, 2)}'
    os.rename(filename, filename.replace('.txt', f'{suffix}.txt'))
    print(f"avg: {np.mean([metric[1] for metric in metrics])}")
    print(f"stdev: {np.std([metric[1] for metric in metrics])}")
    os.rename(args.dir_path, args.dir_path + suffix)

if __name__ == "__main__":
    main()