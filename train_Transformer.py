#!/usr/bin/env python
# coding: utf-8

# In[123]:


from model.Transformer import Transformer
from model.Mango import Mango
import torch
import torchvision.transforms as transforms
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

TRAIN_DIR = "./data/C1-P1_Train/"
TRAIN_CSV = "./data/train.csv"
DEV_DIR = "./data/C1-P1_Dev/"
DEV_CSV = "./data/dev.csv"
Mango_Class = {'A': 0, 'B': 1, 'C': 2}

# transformer
ninp = 3 # input dim
nemb = 128 # embedding dim
nhead = 8 # head num
nhid = 2048 # hidden layer dim
nlayers = 6 # Nx
nclass = 3

# hyper parameters
BS_PER_GPU = 20
NUM_CHANNELS = 3
NUM_CLASSES = 3
NUM_EPOCHS = 100
NUM_ITERS = 100
NUM_TEST_FRE = 1
LR = 0.001
SCH_SETPSIZE = 2
SCH_DECAY = 0.95
IMG_SIZE = 500


# # DataLoader

# In[124]:


class Color:
    def __init__(self, brightness=1, contrast=2):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, imgs):
        imgs = transforms.functional.adjust_brightness(imgs, self.brightness)
        imgs = transforms.functional.adjust_contrast(imgs, self.contrast)
        return imgs


# In[125]:


train_transform = transforms.Compose([
    Color(0.8, 2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop((int(IMG_SIZE/5*4), int(IMG_SIZE/5*4))),
    transforms.ToTensor(),
    transforms.RandomErasing(0.2),
])

test_transform = transforms.Compose([
    Color(0.8, 2),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop((int(IMG_SIZE/4), int(IMG_SIZE/4))),
    transforms.ToTensor(),
])

trainset = Mango(TRAIN_CSV, TRAIN_DIR, Mango_Class, train_transform)
testset = Mango(DEV_CSV, DEV_DIR, Mango_Class, test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS_PER_GPU, shuffle=True, num_workers=6)
testloader = torch.utils.data.DataLoader(testset, batch_size=BS_PER_GPU, shuffle=True, num_workers=6)


# # Model

# In[4]:


model = Transformer(ninp, nemb, nhead, nhid, nlayers, nclass).cuda()
model.apply(model.init_weights)
# model.load_state_dict(torch.load('./model/weights/Transformer/1.pkl'))

loss_fn = torch.nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optim, SCH_SETPSIZE, SCH_DECAY)


# # Training

# In[ ]:


for epoch in range(NUM_EPOCHS):
    """
    Training
    """
    total_loss = 0
    recall = 0
    model.train()
    for i, data in enumerate(trainloader):
        optim.zero_grad()

        imgs, labels = data['data'].cuda(), data['label'].cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optim.step()
        predicts = outputs.argmax(dim=1)
        recall += recall_score(labels.cpu(), predicts.cpu(), average='weighted', zero_division=0)
        if(i % NUM_ITERS == NUM_ITERS-1):
            print("[{:3d}/{:3d}] iter: {:4d}   loss: {:.6f}   WAR: {:5.2f}%".
                  format(epoch+1, NUM_EPOCHS, i+1, total_loss, recall/(i+1) * 100))
            total_loss = 0
            
    """
    Testing
    """
    if(epoch % NUM_TEST_FRE == NUM_TEST_FRE-1):
        total_loss = 0
        recall = 0
        model.eval()
        for i , data in enumerate(testloader):            
            imgs, labels = data["data"].cuda(), data["label"].cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            predicts = torch.argmax(outputs, dim=1)
            recall += recall_score(labels.cpu(), predicts.cpu(), average='weighted', zero_division=0)
            if(i == 0):
                fusion_matrix = confusion_matrix(labels.cpu(), predicts.cpu(), labels=[0, 1, 2])
            else:
                fusion_matrix += confusion_matrix(labels.cpu(), predicts.cpu(), labels=[0, 1, 2])
        
        print("[Testing] loss: {:.6f}   WAR: {:5.2f}%".format(total_loss, recall/(i+1) * 100))
        ConfusionMatrixDisplay(fusion_matrix).plot()
        plt.show()
    scheduler.step()
    
    torch.save(model.state_dict(), './model/weights/Transformer/'+ str(epoch) + '.pkl')

