
import torch
import os

def main():
    # load the weights
    weights_path = './model_weights/eva02_L_pt_m38m_p14to16.pt'
    # weights_path = './eva02_L_pt_m38m_p14to16.pt'

    assert os.path.exists(weights_path), f'{weights_path} does not exist!'
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    checkpoint = checkpoint['model']
    
    for k,v in checkpoint.items():
        print(k)
        
    print('------------------------')
        
    new_checkpoint = dict()
    for k,v in checkpoint.items():
        if k == 'mask_token' or 'lm_head' in k:
            continue
        else:
            new_name = 'backbone.' + k
            new_checkpoint[new_name] = v
    
    for k,v in new_checkpoint.items():
        print(k, v.shape)
    
    torch.save(new_checkpoint, 'model_weights/eva02_L_pt_m38m_p14to16.pth')
    print('done!')
    
    
if __name__ == '__main__':
    main()






