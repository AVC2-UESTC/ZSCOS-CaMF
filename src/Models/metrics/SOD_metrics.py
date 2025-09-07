import math
import time

import numpy as np
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt

from .base_metric import BaseMetric

from ..utils import resize

EPS = 1e-9
# functional

def is_zero(x: torch.Tensor) -> bool:
    if x.abs() < EPS:
    
        return_value = True
    else:
        return_value = False
    
    return return_value


def precision(pred, label, threshold=128) -> torch.Tensor:
    '''
    Args:
        pred: figure ranging from 0 ~ 255
        labels: label indices {0, 1}
        threshold: 0 ~ 255
    
    Return:
        float
    
    Shape: 
        pred: (1, ..., 1, H, W) or (H, W)
        label: same as above
        
    
    '''
    
    binary_segmented_map = (pred > threshold).int()
    TP = torch.sum(binary_segmented_map * label)
    FP = torch.sum(binary_segmented_map) - TP
    
    precision_value = TP / (TP + FP + EPS)
    
    return precision_value

def recall(pred, label, threshold=128) -> torch.Tensor:
    '''
    Args:
        pred: figure ranging from 0 ~ 255
        labels: label indices {0, 1}
        threshold: 0 ~ 255
    
    Return:
        float
    
    Shape: 
        pred: (1, ..., 1, H, W) or (H, W)
        label: same as above

    '''

    binary_segmented_map = (pred > threshold).int()
    TP = torch.sum(binary_segmented_map * label)
    FN = torch.sum(label) - TP
    
    recall_value = TP / (TP + FN + EPS)
    
    return recall_value


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate the ssim score.
        
        Args:
            pred: figure ranging from 0 ~ 1
            labels: label indices {0, 1}
            
        Shape: 
            pred: (1, ..., 1, H, W) or (H, W)
            label: same as above
        """
        h, w = pred.shape[-2:]
        L = h * w

        x = torch.mean(pred)
        y = torch.mean(gt)

        sigma_x = torch.sum((pred - x) ** 2) / (L - 1)
        sigma_y = torch.sum((gt - y) ** 2) / (L - 1)
        sigma_xy = torch.sum((pred - x) * (gt - y)) / (L - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        # if alpha != 0:
        if not is_zero(alpha):
            score = alpha / (beta + EPS)
        elif is_zero(alpha) and is_zero(beta):
            score = 1
        else:
            score = 0
        return score
 



class Precision(object):
    '''
    
    '''

    def __init__(self, resize_logits:bool = False, 
                 # metric args
                 threshold: float = 0.5) -> None:
        self.resize_logits = resize_logits
        self.threshold = threshold

    def __call__(self, preds, labels):
        pass


class Recall(object):
    '''

    '''

    def __init__(self, resize_logits:bool = False, 
                 # metric args
                 threshold: float = 0.5) -> None:
        self.resize_logits = resize_logits
        self.threshold = threshold

    def __call__(self, preds, labels):
        pass



class Fmeasure(object):
    """
    Args:
        resize_logits: ranging 0 ~ 1
        beta:
        mode:
            support {'max', 'mean', 'adaptive'}
    
    """

    def __init__(self, resize_logits: bool = False, 
                 #metric args
                 beta_sq: float = 0.3, 
                 mode: str = 'max'):
        self.fm_list = []
        
        self.resize_logits = resize_logits
        self.beta_sq = beta_sq
        self.mode = mode
        self.init_value = 0.0
      
    @staticmethod  
    def _get_adaptive_threshold(matrix: torch.Tensor, max_value: float = 1) -> float:
        """
        Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
        :param matrix: a data array
        :param max_value: the upper limit of the threshold
        :return: min(2 * matrix.mean(), max_value)
        
        Shape:
            in: (1, ..., 1, H, W) or (H, W)
            
        """
        
        threshold = min(2 * matrix.mean(), max_value)
        
        return threshold
    
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape
        
        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            
        if self.mode == 'max':
            pred = (pred * 255).int() # 0 ~ 1 -> 0 ~ 255
            F_max = torch.tensor(0)
            
            if batch_size == 1:
                for threshold in range(255):
                    
                    P = precision(pred, gt, threshold=threshold)
                    R = recall(pred, gt, threshold=threshold)
                    F_i = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                    
                    if F_i > F_max:
                        F_max = F_i

                self.fm_list.append(F_max.item())
            else:
                for i in range(batch_size):
                    
                    for threshold in range(255):
                        P = precision(pred[i], gt[i], threshold=threshold)
                        R = recall(pred[i], gt[i], threshold=threshold)
                        F_i = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                        if F_i > F_max:
                            F_max = F_i
                    self.fm_list.append(F_max.item())
        
        elif self.mode == 'mean':
            pred = (pred * 255).int() # 0 ~ 1 -> 0 ~ 255
            F = []
            if batch_size == 1:
                for threshold in range(255):
                    
                    P = precision(pred, gt, threshold=threshold)
                    R = recall(pred, gt, threshold=threshold)
                    F_i = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                    
                    F.append(F_i)
                mean_F = torch.mean(torch.stack(F, dim=0))
                self.fm_list.append(mean_F.item())  
            else:
                for i in range(batch_size):
                    for threshold in range(255):
                        P = precision(pred[i], gt[i], threshold=threshold)
                        R = recall(pred[i], gt[i], threshold=threshold)
                        F_i = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                        F.append(F_i)
                    mean_F = torch.mean(torch.stack(F, dim=0))
                    self.fm_list.append(mean_F.item())
                             
        elif self.mode == 'adaptive':
            if batch_size == 1:
                threshold = self._get_adaptive_threshold(pred)
                P = precision(pred, gt, threshold=threshold)
                R = recall(pred, gt, threshold=threshold)
                F_mean = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                self.fm_list.append(F_mean.item())
            else:
                for i in range(batch_size):
                    threshold = self._get_adaptive_threshold(pred[i])
                    P = precision(pred[i], gt[i], threshold=threshold)
                    R = recall(pred[i], gt[i], threshold=threshold)
                    F_mean = (1 + self.beta_sq) * P * R / (self.beta_sq * P + R + EPS)
                    self.fm_list.append(F_mean.item())
        else:
            raise Exception('Unknown mode: {}'.format(self.mode))
                    
                    
        
                    
    
    def compute(self):
        mean_fm = sum(self.fm_list) / len(self.fm_list)
        self.fm_list = []
        return mean_fm
   
   
class WeightedFmeasure(object):
    """
    Args:
        resize_logits: 
        beta:
       
    
    """

    def __init__(self, resize_logits: bool = False, 
                 #metric args
                 beta_sq: float = 0.3, 
                 ):
        self.wfm_list = []
        
        self.resize_logits = resize_logits
        self.beta_sq = beta_sq
        self.init_value = 0.0
        
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
            
        Shape: 
            pred: (N, 1, H, W)
            gt: (N, 1, H, W)
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape

        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
        if batch_size == 1:
            gt = gt.squeeze(0)
            pred = pred.squeeze(0)
            
            wfm = self._cal_wfm(pred, gt)
            self.wfm_list.append(wfm.item())
        else:
            for i in range(batch_size):
                wfm = self._cal_wfm(pred[i], gt[i])
                self.wfm_list.append(wfm.item())
       
                
    def compute(self):
        mean_wfm = sum(self.wfm_list) / len(self.wfm_list)
        self.wfm_list = []
        return mean_wfm
                            
    def _cal_wfm(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Shape:
            pred: (1, H, W)
            gt: (1, H, W)
        '''
        pred = pred.squeeze(0)
        gt = gt.squeeze(0)
        
        gt_cpu = gt.cpu()
        Dst, Idxt = distance_transform_edt(1 - gt_cpu, return_indices=True)
        # ndarray
        Dst = torch.from_numpy(Dst).to(pred.device)
        Idxt = torch.from_numpy(Idxt).to(pred.device)
        
        
        # start_time = time.time()
            
        E = torch.abs(pred - gt).to(dtype=torch.float32)
        Et = E.detach().clone()
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]
        
        K = gauss2D((7, 7), sigma=5, device=pred.device)
        K = torch.rot90(K, k=2, dims=[0, 1])
        
        Et = Et.unsqueeze(0).unsqueeze(0)
        K = K.unsqueeze(0).unsqueeze(0)
        EA = F.conv2d(input=Et, weight=K, padding='same')
        EA = EA.squeeze(0).squeeze(0)
        
        # MIN_E_EA = np.where(gt & (EA < E), EA, E)
        MIN_E_EA = torch.where((gt == 1) & (EA < E), EA, E)
        
        B = torch.where(gt == 0, 2 - torch.exp(torch.log(torch.tensor(0.5)) / 5 * Dst), torch.ones_like(gt))
        Ew = MIN_E_EA * B
        
        #TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        # FPw = np.sum(Ew[gt == 0])
        TPw = torch.sum(gt) - torch.sum(Ew[gt == 1])
        FPw = torch.sum(Ew[gt == 0])
        
        # R = 1 - np.mean(Ew[gt == 1])
        # P = TPw / (TPw + FPw + _EPS)
        
        # R = 1 - torch.nanmean(Ew[gt == 1])
        
        m = torch.mean(Ew[gt == 1])
        if torch.isnan(m):
            R = 1
        else:
            R = 1 - m
        
        P = (TPw + EPS) / (TPw + FPw + EPS)
        # print("R: ", R)
        Q = ((1 + self.beta_sq) * R * P + EPS) / (R + self.beta_sq * P + EPS)
        # time_e = time.time() - start_time
        # print("time_e: ", time_e)
        return Q 
            
            
            
            
            
            
# from matplotlib import pyplot as plt        
            
class Smeasure(object):
    '''
    Args:
        resize_logits: bool
    '''
    def __init__(self, 
                 resize_logits: bool = False, 
                 
                 # metrics args
                 alpha:float = 0.5,
                 
                ) -> None:
        self.resize_logits = resize_logits
        self.alpha = alpha
        self.value_list = []
        
    def update(self, preds: Dict, gts: Dict) -> None:
        '''
        Args:
            preds: Dict[Tensor]
            gt: same 
            
        Shape:
            preds: (N, 1, H, W)
            gts: (N, 1, H, W)
        '''
        
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach().to(torch.float32)
        gt = gt.detach().to(torch.float32)
        batch_size, c, h, w = gt.shape 
        
        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )            
            
        # pred = gt
        if batch_size == 1:
            pred = pred.squeeze(0)
            gt = gt.squeeze(0)
            sm = self._cal_sm(pred, gt)
            # if 1.0 - sm >1e-6:
            #     print(sm)
            self.value_list.append(sm.item())
        else:
            for i in range(batch_size):
                sm = self._cal_sm(pred[i], gt[i])
                self.value_list.append(sm.item())
    
    
    def compute(self):
        mean_sm = sum(self.value_list) / len(self.value_list)
        self.value_list = []
        return mean_sm   
            
    
    def _cal_sm(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(0)
        gt = gt.squeeze(0)
        
        y = torch.mean(gt)
        if is_zero(y):
            sm = 1 - torch.mean(pred)
        elif (y - 1).abs() < EPS:
            sm = torch.mean(pred)
        else:
            region_score = self._region(pred, gt) * (1 - self.alpha)
            object_score = self._object(pred, gt) * self.alpha
            
            # print(object_score, region_score)
            sm = torch.maximum(torch.tensor(0., device=gt.device), object_score + region_score)
            
        # if torch.isnan(sm):
        #     print(sm)
        return sm
            
    
    def _object(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        
        u = torch.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score
    
    def s_object(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        obj = pred[gt == 1]
        # check whether empty
        if obj.numel() == 0:
            score = torch.tensor(0.0, device=obj.device)
        else:
            x = torch.mean(obj)
            sigma_x = torch.std(obj)
            score = 2 * x / (torch.pow(x, 2) + 1 + sigma_x + EPS)
        return score
    
    
    def _region(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        
        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)
        
        # if pred1.numel() == 0 or pred2.numel() == 0 or pred3.numel() == 0 or pred4.numel() == 0:
        #     print("Error: pred1, pred2, pred3, pred4 should not be empty")
        
        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4
        
        # ===========================================================================
        # h, w = gt.shape
        # area = h * w

        # # Calculate the centroid coordinate of the foreground
        # if torch.count_nonzero(gt) == 0:
        #     cy, cx = torch.round(h / 2), torch.round(w / 2)
        # else:
        #     # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
        #     cy, cx = torch.argwhere(gt).mean(dim=0, dtype=torch.float64).round()
        # # To ensure consistency with the matlab code, one is added to the centroid coordinate,
        # # so there is no need to use the redundant addition operation when dividing the region later,
        # # because the sequence generated by ``1:X`` in matlab will contain ``X``.
        # cy, cx = int(cy) + 1, int(cx) + 1

        # # Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        # w_lt = cx * cy / area
        # w_rt = cy * (w - cx) / area
        # w_lb = (h - cy) * cx / area
        # w_rb = 1 - w_lt - w_rt - w_lb
        # score_lt = self.ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        # score_rt = self.ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        # score_lb = self.ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        # score_rb = self.ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        # return score_lt + score_rt + score_lb + score_rb
        
    def centroid(self, matrix: torch.Tensor) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = torch.count_nonzero(matrix)
        if area_object == 0:
            x = torch.round(w / 2)
            y = torch.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = torch.argwhere(matrix).mean(dim=0, dtype=torch.float32).round()
        return int(x) + 1, int(y) + 1
    
    def divide_with_xy(self, pred: torch.Tensor, gt: torch.Tensor, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        out = dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )
        
        # for i in out['gt']:
        #     if i.numel() == 0:
        #         print(i)
        
        return out
    

    def ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = torch.mean(pred)
        y = torch.mean(gt)

        sigma_x = torch.sum((pred - x) ** 2) / (N - 1)
        sigma_y = torch.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = torch.sum((pred - x) * (gt - y)) / (N - 1)
        
        
        
        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if not is_zero(alpha):
            score = (alpha + EPS) / (beta + EPS)
        elif is_zero(alpha) and is_zero(beta):
            score = torch.tensor(1.0, device=pred.device)
        else:
            score = torch.tensor(0.0, device=pred.device)
            
        if pred.numel() == 0 or gt.numel() == 0:
            score = torch.tensor(0.0, device=pred.device)
            
        # if torch.isnan(score):
        #     print(score)
        return score
    
            


 
 

class Emeasure(object):
    def __init__(self):
        """
        E-measure(Enhanced-alignment Measure) for SOD.
        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38
        ::
            @inproceedings{Emeasure,
                title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
                author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
                booktitle=IJCAI,
                pages="698--704",
                year={2018}
            }
            
        Args:
            resize_logits: ranging 0 ~ 1
            mode:
                support 'max' and 'mean'
    
    """

    def __init__(self, resize_logits: bool = False, 
                 #metric args
                 mode: str = 'max'):
        self.em_list = []
        
        self.resize_logits = resize_logits
        self.mode = mode
        self.init_value = 0.0
 
    
    @staticmethod  
    def _get_adaptive_threshold(matrix: torch.Tensor, max_value: float = 1) -> float:
        """
        Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
        :param matrix: a data array
        :param max_value: the upper limit of the threshold
        :return: min(2 * matrix.mean(), max_value)
        
        Shape:
            in: (1, ..., 1, H, W) or (H, W)
            
        """
        
        threshold = min(2 * matrix.mean(), max_value)
        
        return threshold   
    
    
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape
        
        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            
        if self.mode == 'max':
            pred = (pred * 255).int() # 0 ~ 1 -> 0 ~ 255
            E_max = torch.tensor(0)
            
            if batch_size == 1:
                for threshold in range(255):
                    binary_segmented_map = (pred > threshold).int()
                    em_i = self._cal_em(binary_segmented_map, gt)
                    E_max = torch.max(em_i, E_max)
                self.em_list.append(E_max.item())
            else:
                for i in range(batch_size):
                    for threshold in range(255):
                        binary_segmented_map_i = (pred[i] > threshold).int()
                        em_i = self._cal_em(binary_segmented_map_i, gt[i])
                        E_max = torch.max(em_i, E_max)
                    self.em_list.append(E_max.item())
                        
        elif self.mode == 'mean':
            pred = (pred * 255).int() # 0 ~ 1 -> 0 ~ 255
            E = []
            
            if batch_size == 1:
                for threshold in range(255):
                    binary_segmented_map = (pred > threshold).int()
                    em_i = self._cal_em(binary_segmented_map, gt)
                    E.append(em_i)
                E = torch.stack(E)
                self.em_list.append(E.mean().item())
            else:
                for i in range(batch_size):
                    for threshold in range(255):
                        binary_segmented_map_i = (pred[i] > threshold).int()
                        em_i = self._cal_em(binary_segmented_map_i, gt[i])
                        E.append(em_i)
                E = torch.stack(E)
                self.em_list.append(E.mean().item())
             
        elif self.mode == 'adaptive':
            if batch_size == 1:
                threshold = self._get_adaptive_threshold(pred)
                binary_segmented_map = (pred > threshold).int()
                em_i = self._cal_em(binary_segmented_map, gt)
                self.em_list.append(em_i.item())
            else:
                for i in range(batch_size):
                    threshold = self._get_adaptive_threshold(pred[i])
                    binary_segmented_map_i = (pred[i] > threshold).int()
                    em_i = self._cal_em(binary_segmented_map_i, gt[i])
                    self.em_list.append(em_i.item())
                    

    def compute(self):
        mean_E = sum(self.em_list) / len(self.em_list)
        self.em_list = []
        return mean_E
        
        
    def _cal_em(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            pred: binary foreground map after thresholding. Ranging: 0 ~ 1
            gt: binary. {0, 1}
        
        
        Shape:
            pred: (1, ..., 1, H, W) or (H, W)
            gt: same as above
        '''
        mu_GT = torch.mean(gt, dtype=torch.float32)
        mu_FM = torch.mean(pred, dtype=torch.float32)
        phi_GT = gt - mu_GT * torch.ones(gt.shape, dtype=torch.float32, device=gt.device)
        phi_FM = pred - mu_FM * torch.ones(pred.shape, dtype=torch.float32, device=pred.device)
        
        numerator = 2 * phi_GT * phi_FM
        denominator = torch.pow(phi_GT, 2) + torch.pow(phi_FM, 2)
        eps_FM = (numerator + EPS) / (denominator + EPS)
        
        E_align = (1/4) * torch.pow((1 + eps_FM), 2)
        Q_FM = torch.mean(E_align)
        
        return Q_FM

    
    
    

class BER(BaseMetric):
    def __init__(self, resize_logits = False, 
                 #metric args
                 mode = 'BER',
                 threshold: str = 0.5):
        super().__init__(resize_logits)
        self.threshold = threshold
        self.mode = mode
        
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
            
        Shape: 
            pred: (N, 1, H, W)
            gt: (N, 1, H, W)
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape

        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
        
        ber = self._cal_ber(pred, gt)
        self.value_list.extend(ber.tolist())
        
    def _cal_ber(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise ValueError('threshold should not be None')
        else:
            gt = (gt > self.threshold)
            pred = (pred > self.threshold)
            gt = gt.float()
            pred = pred.float()
            TP = pred * gt
            FP = pred * (1 - gt)
            FN = (1 - pred) * gt
            TN = (1 - pred) * (1 - gt)
            
            TP = TP.sum(dim=(1, 2, 3))
            FP = FP.sum(dim=(1, 2, 3))
            FN = FN.sum(dim=(1, 2, 3))
            TN = TN.sum(dim=(1, 2, 3))
            # ber = 0.5*(FP/(TN+FP+EPS) + FN/(FN+TP+EPS))
            #(b, )
            
            pos_err = (1 - TP / (TP + FN + EPS)) * 100
            neg_err = (1 - TN / (TN + FP + EPS)) * 100
            
            if self.mode == 'BER':
                ber = 0.5 * (pos_err + neg_err)
            elif self.mode == 'Shad':
                ber = pos_err
            elif self.mode == 'NoShad':
                ber = neg_err
            else:
                raise ValueError('mode should be BER, Shad or NoShad')

        return ber
    
    

class MAE(object):
    """

    """

    def __init__(self, resize_logits: bool = False, ):
        self.maes = []
        self.resize_logits = resize_logits
        
        self.init_value = 1.0
        
    def update(self, preds: Dict, gts: Dict):
        '''
        
        Shape: 
            in: (N, 1, H, W)    
            
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape
        
        
        
        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
        
        if batch_size == 1:
            mae = torch.abs(pred - gt).mean()
            
            self.maes.append(mae.item())
        else:
            for i in range(batch_size):
                mae = torch.abs(pred[i] - gt[i]).mean()
                self.maes.append(mae.item())
                
    def compute(self):
        mean_mae = sum(self.maes) / len(self.maes)
        self.maes = []
        return mean_mae
                
    
    
    
    
    
    
# ================== utils ========================


def gauss2D(kernel_shape: tuple = (7, 7), sigma: int = 5, device='cpu') -> torch.Tensor:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        
        Shape:
            out:
                h: kernel_shape
        """
        m, n = [(ss - 1) / 2 for ss in kernel_shape]
        y, x = torch.meshgrid(torch.arange(-m, m + 1, device=device), torch.arange(-n, n + 1, device=device))
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    

def centroid(matrix: torch.Tensor) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = torch.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            # non_zero_index = torch.nonzero(matrix)
            # y, x = torch.mean(non_zero_index, dim=0, dtype=torch.float32).round()
            y, x = torch.argwhere(matrix).mean(dim=0, dtype=torch.float32).round()
        return int(x) + 1, int(y) + 1




# ============================================= Deprecated ===================================================

class Smeasure_old(object):
    '''
    Args:
        resize_logits: ranging 0 ~ 1
    '''
    
    def __init__(self, 
                    resize_logits: bool = False,
                    #metric args
                    alpha: float = 0.5) -> None:
        self.resize_logits = resize_logits
        self.alpha = alpha
        self.sm_list = []
        
        self.init_value = 0.0
        
        
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
            
        Shape:
            pred: (N, 1, H, W)
            gt: (N, 1, H, W)    
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape
        
        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )            
            
        if batch_size == 1:
            pred = pred.squeeze(0)
            gt = gt.squeeze(0)
            sm = self._cal_sm(pred, gt)
            self.sm_list.append(sm.item())
        else:
            for i in range(batch_size):
                sm = self._cal_sm(pred[i], gt[i])
                self.sm_list.append(sm.item())
        
    def compute(self):
        mean_sm = sum(self.sm_list) / len(self.sm_list)
        self.sm_list = []
        return mean_sm
        
        
    def _cal_sm(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Shape:
            pred: (1, H, W)
            gt: (1, H, W)
        
        '''     
        pred = pred.to(torch.float32) 
        gt = pred.to(torch.float32)  
        pred = pred.squeeze(0)
        gt = gt.squeeze(0)
        
        y = torch.mean(gt)
        if y == 0:
            sm = 1 - torch.mean(pred)
        elif y == 1:
            sm = torch.mean(pred)
        else:
            sm = self.alpha * self._obj_score(pred, gt) + (1 - self.alpha) * self._region_score(pred, gt)
            sm = torch.maximum(torch.tensor(0, device=sm.device), sm)
            
        return sm
    
    def _obj_score(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        
        u = torch.mean(gt)
        
        score = u * self._s_object(fg, gt) + (1 - u) * self._s_object(bg, 1 - gt)
        return score
    
    
    def _region_score(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        x, y = centroid(gt)
        part_info = self._divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        
        score1 = ssim(pred1, gt1)
        score2 = ssim(pred2, gt2)
        score3 = ssim(pred3, gt3)
        score4 = ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4
    
    @staticmethod
    def _s_object(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        x = torch.mean(pred[gt == 1])
        if torch.isnan(x):
            x = torch.tensor([0], dtype=pred.dtype, device=pred.device)
            
        sigma_x = torch.std(pred[gt==1])
        if torch.isnan(sigma_x):
            sigma_x = torch.tensor([0], dtype=pred.dtype, device=pred.device)
            
        score = 2 * x / (torch.pow(x, 2) + 1 + sigma_x + EPS)
        return score
        
        
    @staticmethod
    def _divide_with_xy(pred: torch.Tensor, gt: torch.Tensor, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w
        
        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )


