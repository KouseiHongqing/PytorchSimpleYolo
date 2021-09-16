'''
函数说明: 
Author: hongqing
Date: 2021-09-16 10:39:17
LastEditTime: 2021-09-16 16:15:23
'''
import numpy as np
import torch
# boxes=torch.Tensor([[100,100,210,210,0.72],
#         [250,250,420,420,0.8],
#         [220,220,320,330,0.92],
#         [100,100,210,210,0.72],
#         [230,240,325,330,0.81],
#         [220,230,315,340,0.9]]) 

def NMS(boxes,box_confidence,thre=0.5):
    # _,indices = boxes[:,4].sort(descending=True)
    _,indices = box_confidence.sort(descending=True)
    class Calculator(object):
        def __init__(self) -> None:
            super().__init__()

        def getPruneRes(self,first,boxes,indices,thre):
            self.first = first
            self.indices = indices
            boxes = torch.FloatTensor(boxes)
            self.x = boxes[:,0]
            self.y = boxes[:,1]
            self.xx = boxes[:,2]
            self.yy = boxes[:,3]
            self.scores = boxes[:,4]
            self.areas = (self.xx-self.x)*(self.yy-self.y)
            self.thre = thre
            return torch.tensor(list(map(self.cal,self.indices)),requires_grad=False)

        def cal(self,indice):
            x = torch.clamp(self.x[indice],min=self.x[self.first])
            y = torch.clamp(self.y[indice],min=self.y[self.first])
            xx = torch.clamp(self.xx[indice],max=self.xx[self.first])
            yy = torch.clamp(self.yy[indice],max=self.yy[self.first])
            area = (xx-x).clamp(min=0)*(yy-y).clamp(min=0)
            if(area/(self.areas[indice]+self.areas[self.first]-area)) < self.thre:
                return False
            return True

    calculator = Calculator()
    res=[]
    while(indices.numel()>0):
        first = indices[0]
        res.append(first.item())
        indices = indices[1:]
        if(indices.numel()==0):
            break
        mask = calculator.getPruneRes(first,boxes,indices,thre)
        indices = indices.masked_select(mask)
    return res

