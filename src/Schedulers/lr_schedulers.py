
import math 

import torch







def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)




# def CosineAnnealingLR_Warmup(optimizer,
#                              dataset_len: int, 
#                              epochs: int,  
#                              warmup_epochs=1, 
#                              warmup_factor=1e-3, 
#                              end_factor=1e-6):
#     num_step = dataset_len
#     assert num_step > 0 and epochs > 0
    
#     def f(x):
#         if warmup_epochs > 0 and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
    
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f) 


class CosineAnnealingLR_Warmup():
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 args:dict) -> None:
        self.optimizer = optimizer
        
        self.num_step = args.get('dataset_len', 100)
        self.epochs = args['epochs']
        self.warmup_epochs = args.get('warmup_epochs', 1)
        self.warmup_factor = args.get('warmup_factor', 1e-3)
        self.end_factor = args.get('end_factor', 1e-6)
        
        
    def get_lr_scheduler(self):
        def f(x):
            if self.warmup_epochs > 0 and x <= (self.warmup_epochs * self.num_step):
                alpha = float(x) / (self.warmup_epochs * self.num_step)
                return self.warmup_factor * (1 - alpha) + alpha
            else:
                current_step = (x - self.warmup_epochs * self.num_step)
                cosine_steps = (self.epochs - self.warmup_epochs) * self.num_step
                return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - self.end_factor) + self.end_factor
            
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=f)




if __name__ == '__main__':
    k = CosineAnnealingLR_Warmup(100, 200, warmup_factor=1e-3, end_factor=1e-6)
    



