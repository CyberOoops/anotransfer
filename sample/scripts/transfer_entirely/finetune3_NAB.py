import os
from multiprocessing import Pool
import sys
sys.path.append('../../../')
from torch.cuda import is_available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import anomalytransfer as at
from glob import glob
from utils import run_time

from typing import Sequence, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s [%(levelname)s]] %(message)s')

config = at.utils.config()
CLUSTER_OUTPUT = config.get("CLUSTERING", "output")
EPOCHS = config.getint("CLUSTERING", "epochs")
BASE_EPOCHS = config.getint('TRANSFER_LEARNING', 'base_epochs')
DATA_EPOCHS = config.getint('TRANSFER_LEARNING', 'data_epochs')
INPUT = config.get('TRANSFER_LEARNING', 'input')
BASE_INPUT = config.get('TRANSFER_LEARNING', 'base_input')
OUTPUT = config.get('TRANSFER_LEARNING', 'output')
MODEL_PATH = config.get('TRANSFER_LEARNING', 'model_path')
RATIO = config.getfloat('TRANSFER_LEARNING', 'ratio')
DOWN_SAMPLING_STEP = config.getint('CLUSTERING_PREPROCESSING', 'down_sampling_step')

#RAW_INPUT = config.get("CLUSTERING_PREPROCESSING", "input")
RAW_INPUT = config.get("TRANSFER_LEARNING", "raw_input")


def _get_latent_vectors(x: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(x)
    seq_length = x.shape[1]
    input_dim = x.shape[2]

    model = at.clustering.LatentTransformer(
        seq_length=seq_length, input_dim=input_dim)
    """ model.fit(x, epochs=EPOCHS, verbose=0)
    model.save(os.path.join(OUTPUT, 'model.pt')) """
    model.load_state_dict(torch.load('/home/cnic/projects/AnoTransfer-code/anomalytransfer/out/clustering/clustering/model.pt'))
    model.fit(x, epochs=EPOCHS, verbose=0)
    #model.save(os.path.join(OUTPUT, 'model.pt'))
    return model.transform(x)


def cluster_data(path: str) -> Tuple[str, str]:
    base = None
    data = None
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            data = item_path
        else:
            base = item_path
    if base is None or data is None:
        raise ValueError('Base path or data path not found')
    return base, data

def make_base_model(kpi: at.transfer.data.KPI, epochs: int, cluster):
    kpi.complete_timestamp()
    kpi, _, _ = kpi.standardize()
    model = at.transfer.models.AnomalyDetector()
    if os.path.exists(os.path.join(OUTPUT, '{}_model.pt'.format(cluster))):
        model.load(OUTPUT, '{}_model.pt'.format(cluster))
        model.fit(kpi=kpi.no_labels(), epochs=epochs, verbose=0)
        print('aaaa')
        model.save(OUTPUT, '{}_model.pt'.format(cluster))
    else:
        model.fit(kpi=kpi.no_labels(), epochs=epochs, verbose=0)
        model.save(OUTPUT, '{}_model.pt'.format(cluster))
    return model

def train_test(train_kpi: at.transfer.data.KPI,
               epochs: int,
               test_kpi: at.transfer.data.KPI = None,
               mask: Optional[Sequence] = None,
               **kwargs) -> float:
    model = at.transfer.models.AnomalyDetector()
    if mask is not None:
        model.load_partial(path=kwargs.get('model_path'), name=kwargs.get('base_kpi').name, mask=mask)
        model.freeze(mask)
        model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=0)
        model.unfreeze(mask)
    model.fit(kpi=train_kpi.no_labels(), epochs=epochs, verbose=0)
    if test_kpi is not None and test_kpi.labels is not None:
        anomaly_scores = model.predict(test_kpi, verbose=0)
        results = at.utils.get_test_results(labels=test_kpi.labels,
                                            scores=anomaly_scores,
                                            missing=test_kpi.missing,
                                            use_spot=False)
        at.utils.log_test_results(name=test_kpi.name, results=results)
        return results['f1score']
    else:
        return None


def _ignore_missing(series_list: Sequence, missing: np.ndarray) -> Tuple[np.ndarray, ...]:
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def get_test_results(
        timestamps: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        missing: np.ndarray,
        values: np.ndarray,
        window_size: int = 120,
        **kwargs) -> Dict:
    timestamps = timestamps[window_size - 1:]
    labels = labels[window_size - 1:]
    scores = scores[window_size - 1:]
    missing = missing[window_size - 1:]
    values = values[window_size - 1:]
    adjusted_timestamps, adjusted_labels, adjusted_scores, adjusted_values = _ignore_missing(
        [timestamps, labels, scores, values], missing=missing
    )

    adjusted_scores = at.utils.adjust_scores(
            labels=adjusted_labels, scores=adjusted_scores)
    """ precision, recall, th = precision_recall_curve(adjusted_labels, adjusted_scores, pos_label=1)


    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    arg_max = np.argmax(f1_score)

    best_precision, best_recall, best_f1_score = precision[arg_max], recall[arg_max], f1_score[arg_max]
    threshold = th[arg_max] """
    max_th = np.max(adjusted_scores)+1
    min_th = np.min(adjusted_scores)
    max_f1 = 0.0
    thres = 0.0
    for i in range(0,1000+1):
        now_thresh = (max_th-min_th)/1000*i+min_th
        accuracy, precision, recall, f_score = get_f1_score(now_thresh, adjusted_scores, adjusted_labels)
        if(f_score>max_f1):
            max_f1 = f_score
            thres = now_thresh
    print('min_th{} max_th{} thres{} f_score{}'.format(min_th,max_th,thres,max_f1))
    return max_f1

def get_f1_score(thresh, test_energy, test_labels):
    pred = (test_energy > thresh).astype(int)

    gt = test_labels.astype(int)
    tp = np.sum(pred * gt)
    tn = np.sum((1-pred) * (1-gt))
    fp = np.sum(pred * (1-gt))
    fn = np.sum((1-pred) * gt)
    
    precision = (tp+0.000001) / (tp + fp + 0.000001)
    recall = (tp+0.000001) / (tp + fn + 0.000001)
    accuracy  = (tp+tn+0.000001)/(tp+tn+fp+fn+0.000001)
    f_score = (2 * precision * recall)/ (precision + recall )
    #print(tp,tn,fp,fn,f_score)

    return accuracy, precision, recall, f_score


#用top_k_cluster中的每个base训练一个模型，然后新到的曲线，找到最近的类，finetune然后预测
def main(finetune_num=200):
    print(finetune_num)
    # with torch.cuda.device(torch.device(f"cuda:{finetune_num//200%2}")):
    clusters = os.listdir(BASE_INPUT)
    base_values = []
    base_models = []
    
    for cluster in tqdm(clusters, total=len(clusters)):
        base, data = cluster_data(os.path.join(BASE_INPUT, cluster))
        base_kpi = at.utils.load_kpi(base)
        base_kpi.complete_timestamp()
        base_kpi, _, _ = base_kpi.standardize()

        file_list = os.listdir(data)
        for f in file_list:
            topk_kpi = at.utils.load_kpi(os.path.join(data,f))
            topk_kpi.complete_timestamp()
            topk_kpi, _, _ = topk_kpi.standardize()
            base_model = make_base_model(topk_kpi, BASE_EPOCHS,cluster)
        base_model = make_base_model(base_kpi, BASE_EPOCHS,cluster)
        #这里base model要训练 不能只用baseKPI训练
        base_models.append(base_model)  
    
    clusters_daily = os.listdir(BASE_INPUT)
    for cluster in tqdm(clusters_daily, total=len(clusters_daily)):
        base, data = cluster_data(os.path.join(BASE_INPUT, cluster))

        dt = pd.read_csv(base)
        base_values.append(dt["value"][::DOWN_SAMPLING_STEP])
    
    file_list = at.utils.file_list(RAW_INPUT)
    cluster_values = []
    finetune_values = []
    test_kpis = []
    names = [] 
    pvt = np.loadtxt(os.path.join(BASE_INPUT,'..','pvt.txt'),delimiter=',').reshape(-1,1)
    

    for file in file_list:
        data_kpi = at.utils.load_kpi(file)
        data_kpi.complete_timestamp()
        data_kpi, _, _ = data_kpi.standardize()
        filename = at.utils.filename(file)
        names.append(filename)
    
        # split idx
        ts = data_kpi.timestamps
        ts = ts % (60 * 60 * 24)
        split_idx = np.where(ts < 300)[0]
        _, data_kpi = data_kpi.split_by_idx(split_idx[0], window_size=1)

        # split to [for cluster] and [for finetune]
        ts = data_kpi.timestamps
        ts = ts % (60 * 60 * 24)
        split_idx = np.where(ts < 300)[0]
        cluster_value, finetune_value = data_kpi.split_by_idx(288, window_size=1)
        unuse_value, test_value = data_kpi.split_by_idx(int(len(data_kpi.timestamps)/2)-finetune_num, window_size=1)
        finetune_value, test_value = test_value.split_by_idx(finetune_num, window_size=1)
        

        cluster_values.append(cluster_value.values[::DOWN_SAMPLING_STEP])
        finetune_values.append(finetune_value)
        test_kpis.append(test_value)
    
    
    # get latent var
    base_values = np.asarray(base_values, dtype=np.float32)[..., None]
    base_feature = _get_latent_vectors(base_values)
    

    cluster_values = np.asarray(cluster_values, dtype=np.float32)[..., None]
    for i in range(cluster_values.shape[0]):
        pace_shift, s_min = at.transfer.models.sbd_3(pvt,cluster_values[i])
        #print(values[i][pace_shift:,:].shape)
        cluster_values[i] = np.append(cluster_values[i][pace_shift:,:],cluster_values[i][:pace_shift,:]).reshape(-1,1)
    cluster_feature = _get_latent_vectors(cluster_values)
    print(base_values.shape,cluster_values.shape)

    tmp_result = {name: 0 for name in names}
    #tmp_result["num_of_points"] = finetune_num
    result={}
    for th in range(3,4):
        print('now_th {}'.format(th))
        tmp_result = {name: 0 for name in names}
        for i, (ft, finetune, test_kpi, name) in enumerate(zip(cluster_feature, finetune_values, test_kpis, names)):
            cluster_idx = np.argmin(np.sum((ft - base_feature)**2, axis=1))
            print('name {} belongs to cluster {}'.format(name, cluster_idx))
            base_model = base_models[cluster_idx]
            pace_shift, sbd_value = at.transfer.models.sbd_3(base_values[cluster_idx],cluster_values[i])
            mask = at.transfer.models.find_optimal_mask(sbd_value,
                                                    #threshold=0.3,
                                                    threshold=th / 10,
                                                    less_mask=((1, 1, 1), (1, 1, 1)),
                                                    greater_mask=((1, 1, 0), (0, 1, 1)),)
            if mask is not None:
                base_model.freeze(mask)
                base_model.fit(kpi=finetune.no_labels(), epochs=DATA_EPOCHS, verbose=0)
                base_model.unfreeze(mask)
            else:
                base_model.fit(kpi=finetune.no_labels(), epochs=DATA_EPOCHS, verbose=0)
            anomaly_scores = base_model.predict(test_kpi, verbose=1)
            f1_score = get_test_results(
                        timestamps=test_kpi.timestamps,
                        labels=test_kpi.labels,
                        scores=anomaly_scores,
                        missing=test_kpi.missing,
                        values=test_kpi.values
                    )
            tmp_result[name] = f1_score
            print(f"{i} - {name}")
        result[str(th/10)] = tmp_result
    return result
    

if __name__ == '__main__':

    """ # for num in range(200, 5000, 200):
    #     main(num)
    with Pool(1) as pool:
        results = pool.map(main, range(200, 201, 200))
        # results = pool.map(main, range(200, 201, 200)) """

    results  = main(2000)
    #results = {'0':0.5}
    with open('NAB_result.json','w') as f:
        f.write(json.dumps(results,indent=4))
    """ results = {'0':0.5}
    results = [results]
    final_result = pd.DataFrame(columns=list(results[0].keys()))
    for res in results:
        final_result = final_result.append(res, ignore_index=True)
    
    final_result = final_result.sort_values("num_of_points")
    final_result.to_csv("result.csv", index=False) """
