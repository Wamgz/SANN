from sklearn.metrics import roc_curve, auc
import numpy as np


def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    
    return err, best_th, right_index

def performances_val(map_score_list):
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for map_score in map_score_list:
        count += 1
        score = float(map_score[0])
        label = float(map_score[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr, tpr)
    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_err, val_APCER, val_BPCER, val_ACER, val_threshold


def performances_tpr_fpr(map_score_val_filename):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    scores = []
    labels = []
    for line in lines:
        try:
            record = line.split()
            scores.append(float(record[0]))
            labels.append(float(record[1]))
        except:
            continue

    fpr_list = [0.1, 0.01, 0.001, 0.0001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    return tpr_list


def get_thresholdtable_from_fpr(scores, labels, fpr_list):
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            live_scores.append(float(score))
    live_scores.sort()
    live_nums = len(live_scores)
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        if not live_scores:
            return [0.5]*10
        threshold_list.append(live_scores[i_sample - 1])
    return threshold_list

# feature  -->   [ batch, channel, height, width ]
def plot_save_jpg(x, out_dir, name):
    x = x.data.numpy()
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(x)
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, name + '.jpg'))
    plt.close()

# Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            hack_scores.append(float(score))
    hack_scores.sort()
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] >= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list


def cal_heatmap(x):
    channels, height, width = 0, 0, 0
    
    if len(x.shape) == 3:
        channels, height, width = x.shape
    elif len(x.shape) == 2:
        height, width = x.shape

    heatmap = torch.zeros(height, width)
    if channels > 0:
        for i in range(channels):
            heatmap += torch.pow(x[i,:,:],2).view(height,width)
    else:
        heatmap += torch.pow(x,2).view(height,width)
    
    return heatmap


def tensorboard_add_image_sample(writer, image, predict, identifier, sample_factor):
    rdn = np.random.randint(1, 1000)
    if rdn == 1:
        writer.add_image(' mini_batch: ' + i + ' image', make_grid(image), epoch)
        writer.add_image(' mini_batch: ' + i + ' predict',  make_grid(predict), epoch)


def feature_2_heat_map(log_dir, x, feature1, feature2, feature3, map_x, epoch, mini_epoch):
    out_dir = os.path.join(log_dir, 'heatmap',  'epoch_' + str(epoch) + '_mini_epoch_' + str(mini_epoch))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # original image
    heatmap = cal_heatmap(x[0,:,:,:].cpu())
    plot_save_jpg(heatmap, out_dir, 'visual')
    plot_save_jpg(x[0,:,:,:].permute((1, 2, 0)).cpu(), out_dir, 'origin_visual')

    ## first feature
    heatmap = cal_heatmap(feature1[0,:,:,:].cpu())
    plot_save_jpg(heatmap, out_dir, 'x_Block1_visual')

    ## second feature
    heatmap = cal_heatmap(feature2[0,:,:,:].cpu())
    plot_save_jpg(heatmap, out_dir, 'x_Block2_visual')

    ## third feature
    heatmap = cal_heatmap(feature3[0,:,:,:].cpu())
    plot_save_jpg(heatmap, out_dir, 'x_Block3_visual')

    ## depth map
    heatmap = cal_heatmap(map_x[0,:,:].cpu())    ## the middle frame 
    plot_save_jpg(heatmap, out_dir, 'x_DepthMap_visual')
    plot_save_jpg(map_x[0,:,:].cpu(), out_dir, 'origin_x_DepthMap_visual')


from numpy import ndarray
from sklearn.metrics import accuracy_score


def get_npcer(false_negative: int, true_positive: int):
    return false_negative / (false_negative + true_positive)


def get_apcer(false_positive: int, true_negative: int):
    return false_positive / (true_negative + false_positive)


def get_acer(apcer: float, npcer: float):
    return (apcer + npcer) / 2.0


def get_metrics(pred: ndarray, targets: ndarray):
    negative_indices = targets == 0
    positive_indices = targets == 1

    false_positive = (pred[negative_indices] == 1).sum()
    false_negative = (pred[positive_indices] == 0).sum()

    true_positive = (pred[positive_indices] == 1).sum()
    true_negative = (pred[negative_indices] == 0).sum()

    npcer = get_npcer(false_negative, true_positive)
    apcer = get_apcer(false_positive, true_negative)

    acer = get_acer(apcer, npcer)

    return acer, apcer, npcer


def get_threshold(probs: ndarray, grid_density: int = 10):
    min_, max_ = min(probs), max(probs)
    thresholds = [min_]
    for i in range(grid_density + 1):
        thresholds.append(min_ + (i * (max_ - min_)) / float(grid_density))
    thresholds.append(1.1)
    return thresholds


def eval_from_scores(map_scores):
    scores, targets = [], []
    for map_score in map_scores:
        count += 1
        score = float(map_score[0])
        label = float(map_score[1])  # int(tokens[1])
        scores.append(score)
        targets.append(label)
    scores = np.array(scores)
    targets = np.array(targets).long()

    thrs = get_threshold(scores)
    acc = 0.0
    best_thr = -1
    for thr in thrs:
        acc_new = accuracy_score(targets, scores >= thr)
        if acc_new > acc:
            best_thr = thr
            acc = acc_new
    return get_metrics(scores >= best_thr, targets), best_thr, acc
