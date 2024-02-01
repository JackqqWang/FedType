
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, Dataset
import random
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from my_utils import *
from conformal import *
torch.manual_seed(10)
np.random.seed(10)
# from utils import *
from options import args_parser
args = args_parser()
device = args.device


class PrivateClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(PrivateClassifier, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x)
    

class PublicClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(PublicClassifier, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x)



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def uni_student_ranking_loss(student_logits, teacher_logits, top_k):
    # top_k list of numpy array, each array is a list of indices

    batch_accumulator_loss = 0
    D = student_logits.shape[1]
    for index in range(len(student_logits)):
        bottom_indices = np.array(list(set(range(D)) - set(top_k[index])))
        top_indices = top_k[index]

        # top_logits = torch.gather(teacher_logits[index], 0, top_indices)
        # bottom_logits = torch.gather(teacher_logits[index], 0, bottom_indices)
        top_logits = teacher_logits[index, top_indices.copy()]
        bottom_logits = teacher_logits[index, bottom_indices]

        top_expanded = top_logits.unsqueeze(-1).expand(-1, len(bottom_indices))
        bottom_expanded = bottom_logits.unsqueeze(-2).expand(len(top_indices), -1)

        pairwise_losses = F.relu(top_expanded - bottom_expanded) / (len(top_indices)*len(bottom_indices))

        batch_accumulator_loss += pairwise_losses.sum()

    return batch_accumulator_loss/student_logits.shape[0]



def student_ranking_loss(student_logits, teacher_logits, top_k=None):

    bottom_k = student_logits.shape[1] - top_k
    student_indices = torch.topk(student_logits, top_k, dim=1).indices
    all_indices = set(range(student_logits.shape[1]))
    bottom_indices = torch.tensor([[i for i in all_indices if i not in indices] for indices in student_indices]).to(student_logits.device)
    
    top_logits = torch.gather(teacher_logits, 1, student_indices)
    #top_logits = teacher_logits[student_indices]
    bottom_logits = torch.gather(teacher_logits, 1, bottom_indices)

    top_expanded = top_logits.unsqueeze(-1).expand(-1, -1, bottom_k)
    bottom_expanded = bottom_logits.unsqueeze(-2).expand(-1, top_k, -1)
    
    # Calculate pairwise ranking losses with margin
    # The loss is max(0, margin - (top - bottom)) for each pair
    pairwise_losses = F.relu(top_expanded - bottom_expanded) / (top_k*2)
    
    # Aggregate the losses, could also use pairwise_losses.mean() to average
    loss = pairwise_losses.sum()
    return loss

distillation_criterion = nn.KLDivLoss() 
# metrics = defaultdict(float)
# combined_loss_function = CombinedLoss(alpha=0.5, beta=0.5).to(device)

class LocalUpdate(object):
    # def __init__(self, args, dataset, idxs, logits_from_llm):
    def __init__(self, args, dataset, idxs, large_model, small_model):
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(
            dataset, idxs)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.large_model = large_model
        self.small_model = small_model

    def train_val_test(self, dataset, idxs):

        np.random.shuffle(idxs)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]
        train_set = Subset(dataset, idxs_train)
        test_set = Subset(dataset, idxs_test)
        trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False, drop_last=True)

        return trainloader, testloader
    
    def update_weights(self,global_round):
        # Set mode to train model
        self.large_model.to(device)
        self.small_model.to(device)
        # small_model.to(device)
        epoch_loss = []
        self.T = self.args.temperature
        T = self.T

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            large_model_optimizer = torch.optim.SGD(self.large_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
            small_model_optimizer = torch.optim.SGD(self.small_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            large_model_optimizer = torch.optim.Adam(self.large_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
            small_model_optimizer = torch.optim.Adam(self.small_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        
        # get the targeted label set
        
        small_cmodel = small_ConformalModel(self.args, self.small_model, self.trainloader,\
                                             alpha=0.1, kreg=args.small_kreg, lamda=args.small_lamda, lamda_criterion='size')
        small_cmodel.eval()

        large_cmodel = large_ConformalModel(self.args, self.large_model, self.trainloader,\
                                             alpha=0.1, kreg=self.args.large_kreg, lamda=args.large_lamda, lamda_criterion='size')
        large_cmodel.eval()

            
        for iter in range(self.args.local_ep):

            # train large model
            self.large_model.train()
            self.small_model.eval()
            print('Start train large model...')

            for param in self.small_model.parameters():
                param.requires_grad = False

            for param in self.large_model.parameters():
                param.requires_grad = True

            batch_loss = []
            # for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.dataset == 'fmnist':
                    images = images.repeat(1,3,1,1)
                    # images = images.view(-1, 28*28)
                images, labels = images.to(device), labels.to(device)

                # CE loss large
                large_model_logits = self.large_model(images)
                ce_large_loss = self.criterion(large_model_logits, labels)

                # small model logits
                with torch.no_grad():
                    small_model_logits = self.small_model(images)
                
                # RKD loss, small to large
                with torch.no_grad():
                    _, my_small_S = small_cmodel(images)
                    _, my_large_S = large_cmodel(images)
                similarity_score = cal_sim_set(my_small_S, my_large_S)
                if self.args.verbose and (batch_idx % 10 == 0):

                    print("| Global Round : {} | Local Epoch : {} | batch {}, similarity_score is {}".format(global_round, iter, batch_idx, similarity_score))

                if similarity_score > 0.9:
                    rkd_loss = 0
                else:
                # my_S = [np.arange(2) for _ in range(len(images))]
                    rkd_loss = uni_student_ranking_loss(large_model_logits, small_model_logits, top_k=my_small_S)
                
                # total large loss
                
                total_large_loss = ce_large_loss + 0.1 * rkd_loss
                
                

                large_model_optimizer.zero_grad()
                total_large_loss.backward()
                large_model_optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), total_large_loss.item()))
                    print("|----in local train epoch {}, batch {}: ce_large_loss is {}, rkd_loss is {}, total large model loss is {}\n".format\
                                            (iter, batch_idx, ce_large_loss, rkd_loss, total_large_loss))
                    
                batch_loss.append(ce_large_loss.item())
            # 
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            # train small model
            self.large_model.eval()
            self.small_model.train()

            print('Start train small model...')

            for param in self.small_model.parameters():
                param.requires_grad = True

            for param in self.large_model.parameters():
                param.requires_grad = False

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)
                if args.dataset == 'fmnist':
                    images = images.repeat(1,3,1,1)
                # CE loss small
                small_model_logits = self.small_model(images)
                ce_small_loss = self.criterion(small_model_logits, labels)
                # if args.dataset == 'fmnist':
                #     images = images.repeat(1,3,1,1)
  
                # KD loss, large to small
                with torch.no_grad():
                    large_model_logits = self.large_model(images)
                soft_labels = torch.softmax(large_model_logits / T, dim=1)

                loss_distillation = distillation_criterion(F.log_softmax(small_model_logits / T, dim=1),
                                                    soft_labels)
                
                # total small loss
                total_small_loss = ce_small_loss + loss_distillation
                
                
                small_model_optimizer.zero_grad()
                total_small_loss.backward()
                small_model_optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), total_small_loss.item()))
                    print("|----in local train epoch {}, batch {}: ce small loss is {}, kd loss is {}, total small model loss is {}\n".format\
                                            (iter, batch_idx, ce_small_loss, loss_distillation, total_small_loss))
                    
                batch_loss.append(ce_large_loss.item())

        small_cmodel = small_ConformalModel(self.args, self.small_model, self.trainloader,\
                                             alpha=0.1, kreg=args.small_kreg, lamda=args.small_lamda, lamda_criterion='size')
        small_cmodel.eval()

        large_cmodel = large_ConformalModel(self.args, self.large_model, self.trainloader,\
                                             alpha=0.1, kreg=self.args.large_kreg, lamda=args.large_lamda, lamda_criterion='size')
        large_cmodel.eval()
        
        return self.small_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
    def get_small_model(self):
        return self.small_model
    

    
    def inference(self):

        self.large_model.eval()
        self.large_model.to(device)
        
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.testloader):


                images, labels = images.to(device), labels.to(device)
                if args.dataset == 'fmnist':
                    images = images.repeat(1,3,1,1)

                outputs = self.large_model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
        return accuracy, loss



