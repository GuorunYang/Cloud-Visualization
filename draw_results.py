import os
import json
import argparse
import rclone
import numpy as np
import matplotlib.pyplot as plt

def check_file_path(file_path):
    if file_path.startswith("tos://"):
        with open(".rclone.conf") as f:
            cfg = f.read()
            result = rclone.with_config(cfg).ls(file_path)
            if (result.get('code') == 0) and (result.get('out') == b''):
                return False
            else:
                return True
    else:
        return os.path.exists(file_path)

def download_remote_file(remote_path, local_path = None):
    if local_path is None:
        local_path = remote_path.split("/")[-1]
    with open(".rclone.conf") as f:
        cfg = f.read()
        rclone.with_config(cfg).copy(remote_path, local_path)

def load_eval_json(json_pth):
    with open(json_pth, 'r') as f:
        json_results = json.load(f)
        return json_results

def collect_eval_results(result_dir):
    collect_metrics = [
        ".-999-999.-CLASSAWARE_Overall_AP-0.50",
        ".-999-999.-CLASSAWARE_TYPE-VEHICLE_AP-0.50",
        ".-999-999.-CLASSAWARE_TYPE-PEDESTRIAN_AP-0.50",
        ".-999-999.-CLASSAWARE_TYPE-CYCLIST_AP-0.50",
        ".-999-999.-CLASSAWARE_TYPE-MISC_AP-0.50",
        ".-68-156.-CLASSAWARE_TYPE-VEHICLE_AP-0.50",
        ".156-220.-CLASSAWARE_TYPE-VEHICLE_AP-0.50",
        ".220-300.-CLASSAWARE_TYPE-VEHICLE_AP-0.50",
    ]
    metric_results = {}
    for metric_name in collect_metrics:
        metric_results[metric_name] = []
    result_list = sorted(os.listdir(result_dir))
    # print("Result list: ", result_list)
    epoch_results = {}
    for i, result_name in enumerate(result_list):
        if result_name.startswith("eval") and result_name.endswith(".json"):
            result_pth = os.path.join(result_dir, result_name)
            epoch_id = int((os.path.splitext(result_name)[0]).split('_')[-1])
            epoch_results[epoch_id] = load_eval_json(result_pth)

    epoch_num = len(epoch_results)
    # print("Epoch result keys: ", epoch_results.keys())
    for i in range(1, epoch_num+1):
        for metric_name in collect_metrics:
            if metric_name in epoch_results[i]:
                metric_results[metric_name].append(epoch_results[i][metric_name])
            else:
                print("Metric {} is NOT in epoch {} results".format(metric_name, i))
                metric_results[metric_name].append(0)
    # print("Epoch results: ", epoch_results)
    # print("Metric results: ", metric_results)
    return epoch_results, metric_results

def draw_epoch_results(metric_results, save_dir):
    for metric_name, eval_results in metric_results.items():
        epoch_num = len(eval_results)
        plt.figure(figsize=(10, 8))
        x = range(1, epoch_num + 1)
        plt.plot(x, eval_results, 'o-', color='b', label=metric_name)
        plt.xlabel("Epoch ID")
        plt.ylabel("mAP")
        if metric_name.startswith("."):
            metric_name = metric_name[1:]
        plt.title(metric_name)
        save_pth = os.path.join(save_dir, "{}.png".format(metric_name))
        plt.savefig(save_pth)

def get_best_epoch(metric_results, use_metric = ".-999-999.-CLASSAWARE_Overall_AP-0.50"):
    if use_metric in metric_results:
        eval_results = metric_results[use_metric]
        best_epoch = np.argmax(eval_results) + 1
        print("Best Epoch Number: {}".format(best_epoch))
        for metric_name, metric_array in metric_results.items():
            print("Metric Name: {} , Result : {:.4f} ".format(metric_name, metric_array[best_epoch-1]))
        last_epoch = len(eval_results)
        print("Last Epoch Number: {}".format(last_epoch))
        for metric_name, metric_array in metric_results.items():
            print("Metric Name: {} , Result : {:.4f} ".format(metric_name, metric_array[last_epoch-1]))



def get_model_path(model_id):
    model_path = model_id
    if model_id.startswith("lidar-t-"):
        model_path = os.path.join("tos://perception-experiments/volc_training/mlflow_artifacts", model_id)
    elif model_id.startswith("t-"):
        model_path = os.path.join("tos://perception-experiments/volc_training/mlflow_artifacts", "lidar-" + model_id)
    model_eval_path = os.path.join(model_path, "mlflow")
    if check_file_path(model_eval_path):
        return model_eval_path
    else:
        print("Model path {} does not exist!".format(model_eval_path))
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw epoch evaluation results')
    parser.add_argument('-n', '--name', type=str, default=None, help='Remote or local name of model(Example: t-20230628035015-rdjph)')
    parser.add_argument('-s', '--save', type=str, default="./", help='Save model path')
    parser.add_argument('-v', '--visual', type=str, default="visual", help='Visual path')
    args = parser.parse_args()

    model_eval_path = get_model_path(args.name)  # Get model path
    if model_eval_path is not None:
        local_eval_path = model_eval_path
        # Download remote evaluation results
        if model_eval_path.startswith("tos://"):
            local_eval_path = os.path.join(args.save, model_eval_path.split("mlflow_artifacts/")[-1])
            print("Download eval results from {} to {}".format(model_eval_path, local_eval_path))
            download_remote_file(model_eval_path, local_eval_path)
        # Determine the visual path for evaluation
        if not args.visual.startswith("/"):
            local_visual_path = os.path.join(local_eval_path.rsplit("/", 1)[0], args.visual)
        os.makedirs(local_visual_path, exist_ok=True)
        epoch_results, metric_results = collect_eval_results(local_eval_path)
        draw_epoch_results(metric_results, local_visual_path)
        get_best_epoch(metric_results)
