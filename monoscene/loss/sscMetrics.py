"""
Part of the code is taken from https://github.com/waterljwant/SSC/blob/master/sscMetrics.py
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(
        predict, axis=1
    )  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc


class DepthMetrics:
    """
    Compute standard depth estimation metrics over batches of predictions and targets.
    Metrics:
      - abs_rel: mean absolute relative error
      - sq_rel: mean squared relative error
      - rmse: root mean squared error
      - rmse_log: root mean squared log error
      - a1, a2, a3: threshold accuracies at 1.25, 1.25^2, 1.25^3
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all stored metric values."""
        self.abs_rels = []
        self.sq_rels = []
        self.rmses = []
        self.rmselog = []
        self.a1s = []
        self.a2s = []
        self.a3s = []

    def add_batch(self, pred, target):
        try: 
            if target.is_cuda:
                target = target.detach().cpu().numpy()
        except AttributeError:
            pass

        try:
            if pred.is_cuda:
                pred = pred.detach().cpu().numpy()
        except AttributeError:
            pass
        #valid = np.logical_and(40 > target, target > 0)

        #target = target.detach().cpu().numpy()
        #pred      = pred.detach().cpu().numpy()
        #target = target.cpu()
        #pred = pred.cpu()
        valid = np.logical_and(target > 0, target < 80)  
        
        pred_valid = pred[valid]
        target_valid = target[valid]
        
        # -----------------------------------------
        # zero / negative 값을 모두 eps로 클램핑
        eps = 1e-6
        pred_valid = np.clip(pred_valid,  eps, None)
        # target_valid = np.clip(target_valid, eps, None)
        # -----------------------------------------

        median_scaling_factor = np.median(target_valid) / np.median(pred_valid)
        pred_valid *= median_scaling_factor   

        abs_rel = np.mean(np.abs(pred_valid - target_valid) / target_valid)
        sq_rel  = np.mean((pred_valid - target_valid)**2 / target_valid)
        rmse    = np.sqrt(np.mean((pred_valid - target_valid)**2))
        rmse_log = np.sqrt(np.mean((np.log(pred_valid) - np.log(target_valid))**2))
        rmse_log = np.sqrt(np.mean((np.log(pred_valid) - np.log(target_valid))**2))

        # threshold accuracy
        thresh = np.maximum(pred_valid / target_valid, target_valid / pred_valid)
        a1 = np.mean(thresh < 1.25)
        a2 = np.mean(thresh < 1.25**2)
        a3 = np.mean(thresh < 1.25**3)

        self.abs_rels.append(abs_rel)
        self.sq_rels.append(sq_rel)
        self.rmses.append(rmse)
        self.rmselog.append(rmse_log)
        self.a1s.append(a1)
        self.a2s.append(a2)
        self.a3s.append(a3)

    # def add_batch(self, pred, target):
    #     """
    #     Add a batch of depth predictions and ground truths.

    #     Args:
    #         pred (np.ndarray): Predicted depths, shape (B, H, W)
    #         target (np.ndarray): Ground truth depths, shape (B, H, W)
    #     """
    #     # Flatten and mask invalid depths
    #     valid = target > 0
    #     pred_valid = pred[valid]
    #     target_valid = target[valid]
    #     if pred_valid.size == 0:
    #         return

    #     # Compute errors
    #     abs_rel = np.mean(np.abs(pred_valid - target_valid) / target_valid)
    #     sq_rel = np.mean((pred_valid - target_valid)**2 / target_valid)
    #     rmse = np.sqrt(np.mean((pred_valid - target_valid)**2))
    #     rmse_log = np.sqrt(np.mean((np.log(pred_valid) - np.log(target_valid))**2))

    #     # Threshold accuracies
    #     thresh = np.maximum(pred_valid / target_valid, target_valid / pred_valid)
    #     a1 = np.mean(thresh < 1.25)
    #     a2 = np.mean(thresh < 1.25**2)
    #     a3 = np.mean(thresh < 1.25**3)

    #     # Store
    #     self.abs_rels.append(abs_rel)
    #     self.sq_rels.append(sq_rel)
    #     self.rmses.append(rmse)
    #     self.rmselog.append(rmse_log)
    #     self.a1s.append(a1)
    #     self.a2s.append(a2)
    #     self.a3s.append(a3)

    def get_results(self):
        """
        Return the averaged metrics over all added batches.

        Returns:
            dict: {
                'abs_rel': float,
                'sq_rel': float,
                'rmse': float,
                'rmse_log': float,
                'a1': float,
                'a2': float,
                'a3': float
            }
        """
        # return {
        #     'abs_rel': float(np.mean(self.abs_rels)) if self.abs_rels else 0.0,
        #     'sq_rel': float(np.mean(self.sq_rels)) if self.sq_rels else 0.0,
        #     'rmse': float(np.mean(self.rmses)) if self.rmses else 0.0,
        #     'rmse_log': float(np.mean(self.rmselog)) if self.rmselog else 0.0,
        #     'a1': float(np.mean(self.a1s)) if self.a1s else 0.0,
        #     'a2': float(np.mean(self.a2s)) if self.a2s else 0.0,
        #     'a3': float(np.mean(self.a3s)) if self.a3s else 0.0,
        # }
        return {
            'abs_rel': float(np.mean(self.abs_rels)),
            'sq_rel': float(np.mean(self.sq_rels)),
            'rmse': float(np.mean(self.rmses)),
            'rmse_log': float(np.mean(self.rmselog)),
            'a1': float(np.mean(self.a1s)),
            'a2': float(np.mean(self.a2s)),
            'a3': float(np.mean(self.a3s)),
        }


class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
                y_true = y_true[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum
