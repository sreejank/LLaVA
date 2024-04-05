import pandas as pd 
import numpy as np 
import ast 
import matplotlib.pyplot as plt 
from PIL import Image 
import json
import uuid 
plt.ioff()

def make_grid_image(grid,fname):
    my_dpi=100
    scale=270
    plt.figure(figsize=(scale/my_dpi, scale/my_dpi), dpi=my_dpi)
    plt.imshow(grid,cmap='bwr',vmin=-1,vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname)
    img=Image.open(fname).convert('RGB').resize((224,224))
    img.save(fname)
    img.close()
    plt.close()
    


 

json_lst=[]
from tqdm import tqdm 
def make_sample(grid):
    id=uuid.uuid4()
    path='grid_images/{}.png'.format(id)
    make_grid_image(grid,path)
    s="There are red tiles in the following (x,y) coordinates: "
    for c in list(np.vstack(np.where(grid==1)).T):
        s+="("+str(c[0])+","+str(c[1])+'), '

    conversations=[
        {
        "from":"human",
        "value": "Now list the location of all (x,y) coordinates for which there is a red tile. "
        },
        {
            "from": "gpt",
            "value": s
        }

    ]
    return dict(id=id,image=path,conversations=conversations)

for ex_id in range(150000):
    if ex_id%1000.0==0.0:
        print(ex_id)
    grid=np.random.choice([0,1],size=49).reshape((7,7)).astype('int')
    json_lst.append(make_sample(grid))

for sample in json_lst:
    sample['id']=str(sample['id'])
    sample['image']='/scratch/gpfs/sreejank/LLaVA/'+sample['image']


with open('data.json', 'w') as f:
    json.dump(json_lst, f)