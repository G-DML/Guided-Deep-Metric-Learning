from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset, Subset
from torch.nn.modules.loss import TripletMarginLoss
from itertools import permutations
from datetime import datetime
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def Metrics(y_real, y_pred):
    acc = metrics.accuracy_score(y_real, y_pred)
    prec = metrics.precision_score(y_real, y_pred, average='macro')
    rec = metrics.recall_score(y_real, y_pred, average='macro')
    f = metrics.f1_score(y_real, y_pred, average='macro')

    print("The average scores for all classes:")
    # Calculate metrics for each label, and find their unweighted mean. does not take label imbalance into account.
    print("\nAccuracy:  {:.2f}%".format(acc * 100))  # (TP+TN)/Total / number of classes
    print("Precision: {:.2f}%".format(prec * 100))  # TP/(TP+FP) / number of classes
    print("Recall:    {:.2f}%".format(rec * 100))  # TP/(TP+FN) / number of classes
    print("F-measure: {:.2f}%".format(f * 100))  # 2 * (prec*rec)/(prec+rec) / number of classes

    print("\nThe scores for each class:")
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_real, y_pred)

    print("\n|    Label    |  Precision |  Recall  | F1-Score | Support")
    print("|-------------|------------|----------|----------|---------")
    for i in range(num_classes):
        print(
            f"| {classes[i]:<11} |  {precision[i] * 100:<7.2f}%  | {recall[i] * 100:<7.2f}% |   {fscore[i]:<4.2f}   | {support[i]}")

    return rec

color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
marker = ['.','+','x','1','^','s','p','*','d','X']
classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def Ploting2D(embeddings_plot,labels_plot,tit="default",x_axis="X",y_axis="Y"):
    ax = plt.figure().add_subplot(111)
    for i in range(num_classes):
        index = labels_plot == i
        plt.scatter(embeddings_plot[0, index], embeddings_plot[1, index], s=3, marker='.', c=color[i], label=classes[i])
    ax.legend(loc='best', title="Labels", markerscale=5.0)

    # add grid
    plt.grid(True,linestyle='--')

    # add title
    plt.title(tit)
    plt.tight_layout()

    # add x,y axes labels
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

def Ploting3D(embeddings_plot, labels_plot, tit="default",x_axis="X",y_axis="Y",z_axis="Z"):
    ax = plt.figure().gca(projection='3d')
    for i in range(num_classes):
        index = labels_plot == i
        ax.scatter(embeddings_plot[0, index], embeddings_plot[1, index], embeddings_plot[2, index], s=3, marker='.',c=color[i], label=classes[i])
    ax.legend(loc='best', title="Labels", markerscale=5.0)

    # add title
    plt.title(tit)
    plt.tight_layout()

    # add x,y axes labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)

#####################################################################################################################
print("\nLOAD DATA\n")

mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768) #CIFAR

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std) ])

class TripletDataset(Dataset):
    def __init__(self, dataset, batch_size=2, shuffle=True, transform=None):
        #self.dataset = dataset
        self.Anchor = torch.tensor([])
        self.Positive = torch.tensor([])
        self.Negative = torch.tensor([])
        #self.Anchor = torch.tensor([], dtype=torch.uint8)
        #self.Positive = torch.tensor([], dtype=torch.uint8)
        #self.Negative = torch.tensor([], dtype=torch.uint8)
        self.Labels = []
        self.batch_size = batch_size
        self.transform = transform

        samples, lab_set, min_size = self.split_by_label(dataset)

        self.batch_size = min(self.batch_size, min_size)

        lab_set_perm = list(permutations(lab_set, 2))

        if shuffle:
            np.random.shuffle(lab_set_perm)

        for i, j in lab_set_perm:
            a, p, n = self.Triplets_maker(samples[i], samples[j])

            self.Anchor = torch.cat((self.Anchor, a), 0)
            self.Positive = torch.cat((self.Positive, p), 0)
            self.Negative = torch.cat((self.Negative, n), 0)
            self.Labels += [[i, j] for _ in range(self.batch_size)]

        #print(f"self.Labels: {self.Labels}")
        print(f"Number of labels permutations: {len(lab_set_perm)}")
        print(f"Triplet samples per permutations: {self.batch_size}")
        print(f"Total number of triplet samples: {len(self.Anchor)}")

    def __len__(self):
        return len(self.Anchor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_sample = self.Anchor[idx]
        positive_sample = self.Positive[idx]
        negative_sample = self.Negative[idx]
        landmarks = torch.tensor(self.Labels)[idx]

        if self.transform:
            anchor_sample = self.transform(anchor_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)
            #landmarks = self.transform(landmarks)

        return (anchor_sample, positive_sample, negative_sample), landmarks

    def split_by_label(self, dataset):
        labels_set = list(dataset.class_to_idx.values())

        samples_by_label = {}
        label_size = []
        for label in labels_set:
            samples_by_label[label] = torch.tensor(dataset.data[np.array(dataset.targets) == label])

            l, w, h, f = samples_by_label[label].shape
            label_size.append(l)

            #samples_by_label[label].view(l, f, w, h)
            #samples_by_label[label] = torch.reshape(samples_by_label[label], (l, f, w, h))
            #samples_by_label[label] = np.transpose(samples_by_label[label], (0, 3, 1, 2))
            samples_by_label[label] = samples_by_label[label].permute(0, 3, 1, 2)

        return samples_by_label, labels_set, np.min(label_size) // 2

    def Triplets_maker(self, class_1, class_2):
        index_ap = np.random.choice(range(len(class_1)), self.batch_size * 2, replace=False)
        index_n = np.random.choice(range(len(class_2)), self.batch_size, replace=False)

        anchor = class_1[index_ap[:self.batch_size]]
        positive = class_1[index_ap[self.batch_size:]]
        negative = class_2[index_n]

        return anchor, positive, negative

#Load Data
train_dataset = CIFAR10(root='dataset/', train=True, transform=preprocess, download='True')
test_dataset = CIFAR10(root='dataset/', train=False, transform=preprocess, download='True')

#Data to triplet format
batch_size = 64
triplet_ds = TripletDataset(dataset=train_dataset, batch_size=batch_size)

#Dataset to Batches
triplet_ld = DataLoader(triplet_ds, batch_size=batch_size, shuffle=False)
##################################################### Using a GPU #####################################################
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("USING", device)
if cuda:
    num_dev = torch.cuda.current_device()
    print(torch.cuda.get_device_name(num_dev),"\n")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, data, device, yield_labels=True):
        self.data = data
        self.device = device
        self.yield_labels = yield_labels

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        if self.yield_labels:
            for data, l in self.data:
                yield to_device(data, self.device), l
        else:
            for data in self.data:
                yield to_device(data, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.data)

#Batches to GPU
train_ld = DeviceDataLoader(triplet_ld, device)

#####################################################################################################################
print("\nGEMINI TRAINING\n")

class TripletNet_conv(nn.Module):
    def __init__(self, in_size=2, out_size=2, num_classes=2, m=1.0, beta=0.0, pow=2):
        super().__init__()
        self.beta = beta
        self.convnet = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(), #nn.Conv2d(in_channels, out_channels, kernel_size)
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2)) for _ in range(num_classes)])
        self.fc = nn.Sequential(nn.Linear(1600, 256), #nn.Linear(64 * 4 * 4 = 1024, 256)
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(), #x,if x≥0 | ax, otherwise. where 'a' is a learnable parameter.
                                nn.Linear(256, out_size)
                                )
        self.triplet_loss = nn.TripletMarginLoss(margin=m, p=pow, eps=0) #max{d(a,p)−d(a,n)+margin,0}
        self.p_dist = lambda a,p: torch.mean(nn.PairwiseDistance(p=pow, eps=0, keepdim=False)(a, p))

    def forward(self, x, c, local_out=False):
        local_output = self.convnet[c](x)
        local_output = local_output.view(local_output.size(0), -1)

        global_output = self.fc(local_output)

        if local_out:
            return global_output, local_output

        return global_output

    def training_step(self, A, P, N,labels):
        l1, l2 = labels

        # Generate predictions
        anchor, a = self(A, l1, local_out=True)  #self.forward(A,label,local_out)
        positive, p = self(P, l1, local_out=True)
        negative = self(N, l2)

        # Calculate Loss
        loss = (1.0 - self.beta) * self.triplet_loss(anchor, positive, negative) + self.beta * self.p_dist(a, p)

        return loss

    def get_new_Dataset(self, dataset, device):
        labels_set = list(dataset.class_to_idx.values())

        Dataset = torch.tensor([])
        Embeddings = torch.tensor([]).to(device)
        Labels = torch.tensor([])

        with torch.no_grad():
            for i, c in enumerate(labels_set):
                indexes = np.array(dataset.targets) == c

                data_subset = torch.tensor(dataset.data[indexes])

                #l, w, h, f = data_subset.shape
                data_subset = data_subset.permute(0, 3, 1, 2)

                labels_subset = torch.tensor(dataset.targets)[indexes]

                embedded = self.forward(data_subset.to(device, dtype=torch.float32), i)

                Dataset = torch.cat((Dataset, data_subset), 0)
                Embeddings = torch.cat((Embeddings, embedded), 0)
                Labels = torch.cat((Labels, labels_subset), 0)

        Dataset = Dataset.cpu().numpy()
        Embeddings = Embeddings.cpu().numpy()
        Labels = Labels.t().cpu().numpy()

        del embedded, labels_subset, labels_set
        torch.cuda.empty_cache()  # PyTorch thing

        return (Dataset, Embeddings, Labels)

def fit(epochs, lr, model, train_loader, opt_func=torch.optim.SGD): #fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD)
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        losses = []
        # Training Phase
        for (A,P,N),Label in train_loader:
            label = Label[0].numpy()
            loss = model.training_step(A,P,N,label) #Generate predictions, calculate loss
            losses.append(loss.item())

            loss.backward()  #Compute gradients

            optimizer.step() #Adjust the weights
            optimizer.zero_grad() #Reset the gradients

        loss_mean = sum(losses)/len(losses)
        history.append(loss_mean)

        print(f"Epoch [{epoch+1}/{epochs}], loss: {history[-1]:.5f}")

# Parameters
num_classes = 10
output_dim = 2
epochs = 12
learn_rate = 0.001
margin = 3.0
beta = 0.003

print(f"Output dimension: {output_dim}\n")

# Model (on Device)
model = TripletNet_conv(out_size=output_dim, num_classes=num_classes, m=margin, beta=beta)
model.to(device)

# Train Model
print(f"[{datetime.now()}]")
start = time.time()

fit(epochs=epochs, lr=learn_rate, model=model, train_loader=train_ld)

end = time.time()-start
print(f"[{datetime.now()}]")
print(f"\nTotal time = {int(end//3600):02d}:{int((end//60))%60:02d}:{end%60:.6f}")

#####################################################################################################################
print("\nPLOTTING SPACE\n")

train_samples, target_train_embeddings, train_labels = model.get_new_Dataset(train_dataset, device)
''' Uncomment to plot the embeddings
if output_dim == 2:
    Ploting2D(target_train_embeddings.T, train_labels, "Data Space")
if output_dim == 3:
    Ploting3D(target_train_embeddings.T, train_labels, "Data Space")
'''
#####################################################################################################################
del model, train_ld, triplet_ld
torch.cuda.empty_cache() # PyTorch thing
#####################################################################################################################
print("\nLOAD NEW DATA\n")

class CNN_Dataset(Dataset):
    def __init__(self, dataset, embeddings, labels=None, transform=None):
        self.dataset = dataset
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataset = self.dataset[idx]
        embeddings = self.embeddings[idx]
        if self.labels is not None:
            labels = self.labels[idx]

        if self.transform:
            dataset = self.transform(dataset)
            embeddings = self.transform(embeddings)
            if self.labels is not None:
                labels = self.transform(labels)

        if self.labels is not None:
            return (dataset, embeddings, labels)

        return (dataset, embeddings)  # , labels

train_ds = CNN_Dataset(dataset=train_samples, embeddings=target_train_embeddings, labels=train_labels)

# Create validation & training datasets
val_size = int(len(train_ds) * 0.10)
train_size = len(train_ds) - val_size
train_ds, validation_ds = random_split(train_ds, [train_size, val_size])

print("Train dataset size: ", len(train_ds))
print("Validation dataset size: ", len(validation_ds))

#####################################################################################################################
print("\nResNet18 TRAINING\n")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, output_size=2, pow=2.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, output_size)

        self.p_dist = lambda a, b: torch.mean(nn.PairwiseDistance(p=pow, eps=0, keepdim=False)(a, b))
        self.p_dist_sum = lambda a, b: torch.sum(nn.PairwiseDistance(p=pow, eps=0, keepdim=False)(a, b))
        self.cos_sim = lambda a, b: 1.0 - torch.mean(nn.CosineSimilarity(dim=1, eps=1e-08)(a, b))
        self.gauss_sim = lambda a, b: torch.sum(1.0 - torch.exp(-(nn.PairwiseDistance(p=2.0, eps=0, keepdim=False)(a, b) ** 2) / (2.0 * 0.125)))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch):
        images, embeddings, _ = batch

        # Generate predictions
        out = self(images)

        # Calculate loss
        loss = self.p_dist(out, embeddings)

        return loss

    def extract_embedding(self, dataset):
        size_batch = len(dataset)//30 #change this if not enough memory

        data_loader = DataLoader(dataset=dataset, batch_size=size_batch, shuffle=False)
        data_loader = DeviceDataLoader(data_loader, 'cuda')

        with torch.no_grad():
            self.train()
            embedding = torch.tensor([]).to('cuda')
            for batch in data_loader:
                data, _ = batch
                embedded = self.forward(data)
                embedding = torch.cat((embedding, embedded), 0)

        embedding = embedding.cpu().numpy()  # embedding.t().cpu().numpy()
        label = torch.tensor(dataset.targets).cpu().numpy()

        del data_loader, embedded, data
        torch.cuda.empty_cache()  # PyTorch thing

        return embedding, label

    def evaluate_step(self, val_loader):
        with torch.no_grad():
            self.eval()

            val_loss = []
            #val_acc = []
            for batch in val_loader:
                images, embeddings, labels = batch

                # Generate predictions
                out_embedding = self(images)

                # Calculate loss
                loss = self.p_dist(out_embedding, embeddings)

                val_loss.append(loss.item())

        epoch_loss = torch.tensor(val_loss).mean()  # Combine losses

        return {'val_loss': epoch_loss.item()}

def ResNet_size(size=10, in_channels=3, output_size=2):
    if size == 10:
        return ResNet(BasicBlock, [1, 1, 1, 1], in_channels=in_channels, output_size=output_size)
    elif size == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, output_size=output_size)
    elif size == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, output_size=output_size)
    elif size == 50:
        return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, output_size=output_size)
    elif size == 101:
        return ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, output_size=output_size)
    elif size == 152:
        return ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, output_size=output_size)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0.0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    # history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up learning rate scheduler
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.20, patience=3, verbose=True)

    for epoch in range(epochs):
        model.train()  # tells the model is in training mode, so batchnorm, dropout and all the ohter layer that have a training mode should get to the training mode
        '''
        # Freeze BN layers
        for module in model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
        '''
        train_losses = []
        lrs = []

        # Training Phase
        for batch in train_loader:

            optimizer.zero_grad()  # Reset the gradients

            loss = model.training_step(batch)  # Generate predictions, calculate loss
            train_losses.append(loss.item())

            loss.backward()  # Compute gradients

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()  # Adjust the weights

        # Record & update learning rate
        mean_loss = sum(train_losses) / len(train_losses)
        lrs.append(get_lr(optimizer))
        sched.step(mean_loss)

        # Validation phase
        result = model.evaluate_step(val_loader)
        result['train_loss'] = mean_loss
        result['lrs'] = lrs

        print(f"Epoch [{epoch + 1}/{epochs}], last_lr: {lrs[-1]:.5f}, train_loss: {mean_loss:.4f}, val_loss: {result['val_loss']:.4f}") #,val_acc: {result['val_acc']:.4f}")

# Model (on Device)
resnet_model = to_device(ResNet_size(size=18, in_channels=3, output_size=output_dim), device)

# Batch size
batch_size = 32

#Dataset to Batches
resnet_train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
resnet_val_ld = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

#Batches to GPU
resnet_train_ld = DeviceDataLoader(resnet_train_ld, device, yield_labels=False)
resnet_val_ld = DeviceDataLoader(resnet_val_ld, device, yield_labels=False)

# Parameters
epochs = 50
max_lr = 0.1

grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.SGD

# Train ResNet
print(f"[{datetime.now()}]")
start = time.time()

fit(epochs, max_lr, resnet_model, resnet_train_ld, resnet_val_ld, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)

end = time.time()-start
print(f"[{datetime.now()}]")
print(f"\nTotal time = {int(end//3600):02d}:{int((end//60))%60:02d}:{end%60:.6f}")

#####################################################################################################################
del resnet_train_ld, resnet_val_ld
torch.cuda.empty_cache() # PyTorch thing
#####################################################################################################################
print("\nPLOTTING NEW SPACE\n")

reference_embeddings, reference_labels = resnet_model.extract_embedding(train_dataset)
''' Uncomment to plot the embeddings
if output_dim == 2:
    Ploting2D(reference_embeddings.T, reference_labels, "Learned Data Space")
if output_dim == 3:
    Ploting3D(reference_embeddings.T, reference_labels, "Learned Data Space")
'''
########################################### Evaluation ##############################################################
knn = KNeighborsClassifier(n_neighbors=1) #algorithm auto = ball_tree, kd_tree or brute
knn.fit(reference_embeddings, reference_labels)
#####################################################################################################################
print("\nPLOTTING GENERALIZATION\n")

query_embeddings, query_labels = resnet_model.extract_embedding(test_dataset)
''' Uncomment to plot the embeddings
if output_dim == 2:
    Ploting2D(query_embeddings.T, query_labels, "Learned Data Embedding")
if output_dim == 3:
    Ploting3D(query_embeddings.T, query_labels, "Learned Data Embedding")
'''
########################################### Evaluation ##############################################################
y_pred = knn.predict(query_embeddings)

recall = Metrics(query_labels, y_pred)

calculator = AccuracyCalculator( exclude=("AMI","mean_average_precision"),
                    avg_of_avgs=False,
                    k="max_bin_count",
                    label_comparison_fn=None)

acc_dict = calculator.get_accuracy(query_embeddings, reference_embeddings, query_labels, reference_labels, embeddings_come_from_same_source=False)

print("\nNMI: ", acc_dict["NMI"]*100)
print("p@1: ", acc_dict["precision_at_1"]*100)
print("RP: ", acc_dict["r_precision"]*100)
print("MAP@R: ", acc_dict["mean_average_precision_at_r"]*100)

#####################################################################################################################
#print("\nSAVING MODEL\n")
'''The .state_dict method returns an OrderedDict containing all the weights and bias matrices mapped to the right attributes of the model'''
#File_name = "Log/GEMINI/Cifar10/" + str(output_dim) + "d/Cifar10_" + str(recall) + "_bs-" + str(batch_size) + "_eps-" + str(epochs) + "_lr-" + str(max_lr) + "_m-" + str(margin) + ".pth"
#torch.save(resnet_model.state_dict(), File_name)
#####################################################################################################################
''' Uncomment to plot the embeddings
if output_dim <= 3:
    plt.show()
'''