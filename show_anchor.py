import torch
import sys
sys.path.append(".")
weights = 'runs/exp_3_15/weights/best.pt'
model = torch.load(str(weights[0] if isinstance(weights, list) else weights), map_location='cpu')
model1 = model['ema' if model.get('ema') else 'model']
model2 = model1.float().fuse().model.state_dict()

for k,v in model2.items():
    if 'anchor' in k:
        # print(k)
        # print(v)
        print(v.numpy().flatten().tolist())
