from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import GradScaler

# 使用CUDA_VISIBLE_DEVICES指定gpu --nproc_per_node=2 用2卡
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --use_mix_precision


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

####################################    N11    ##################################
def evaluate(model, gpu, test_loader, rank):
    model.eval()
    size = torch.tensor(0.).to(gpu)
    correct = torch.tensor(0.).to(gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    # 群体通信 reduce 操作 change to allreduce if Gloo
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    # 群体通信 reduce 操作 change to allreduce if Gloo
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Evaluate accuracy is {:.2f}'.format(correct / size))
 #################################################################################

def train(gpu, args):
    ##################################################################
    # 训练函数中仅需要更改初始化方式即可。在ENV中只需要指定init_method='env://'。
    # TCP所需的关键参数模型会从环境变量中自动获取，环境变量可以在程序外部启动时设定，参考启动方式。
    # 当前进程的rank值可以通过dist.get_rank()得到
    dist.init_process_group(backend='nccl', init_method='env://')    #
    args.rank = dist.get_rank()                                      #
    ##################################################################
    model = ConvNet()
    model.cuda(gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    #######################################    N2    ########################
    # 并行环境下，对于用到BN层的模型需要转换为同步BN层；
    # 用DistributedDataParallel将模型封装为一个DDP模型，并复制到指定的GPU上。
    # 封装时不需要更改模型内部的代码；设置混合精度中的scaler，通过设置enabled参数控制是否生效。
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)                  #
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])    #
    scaler = GradScaler(enabled=args.use_mix_precision)                     #
    #########################################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./datasets',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    ####################################    N3    #######################################
    # DDP要求定义distributed.DistributedSampler，通过封装train_dataset实现；在建立DataLoader时指定sampler。
    # 此外还要注意：shuffle=False。DDP的数据打乱需要通过设置sampler，参考N4。
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)      #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,                   #
                                               batch_size=args.batch_size,              #
                                               shuffle=False,                           #
                                               num_workers=0,                           #
                                               pin_memory=True,                         #
                                               sampler=train_sampler)                   #
    #####################################################################################
    ####################################    N9    ###################################
    test_dataset = torchvision.datasets.MNIST(root='./datasets',                        #
                                              train=False,                          #
                                              transform=transforms.ToTensor(),      #
                                              download=True)                        #
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)    #
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,                 #
                                              batch_size=args.batch_size,           #
                                              shuffle=False,                        #
                                              num_workers=0,                        #
                                              pin_memory=True,                      #
                                              sampler=test_sampler)                 #
    #################################################################################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        ################    N4    ################
        # 在每个epoch开始前打乱数据顺序。（注意total_step已经变为orignal_length // args.world_size。） 
        train_loader.sampler.set_epoch(epoch)    #
        ##########################################
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            ########################    N5    ################################
            # 利用torch.cuda.amp.autocast控制前向过程中是否使用半精度计算。
            with torch.cuda.amp.autocast(enabled=args.use_mix_precision):    #
                outputs = model(images)                                      #
                loss = criterion(outputs, labels)                            #
            ##################################################################

            # Backward and optimize
            optimizer.zero_grad()
            ##############    N6    ##########
            # 当使用混合精度时，scaler会缩放loss来避免由于精度变化导致梯度为0的情况。
            scaler.scale(loss).backward()    #
            scaler.step(optimizer)           #
            scaler.update()                  #
            ##################################
            ################    N7    ####################
            # 为了避免log信息的重复打印，可以只允许rank0号进程打印。
            if (i + 1) % 100 == 0 and args.rank == 0:    #
                ##############################################
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
    ###########    N8    ############
    # 清理进程
    dist.destroy_process_group()    #
    if args.rank == 0:              #
        #################################
        print("Training complete in: " + str(datetime.now() - start))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpuid', default=0, type=int,
                        help="which gpu to use")
    parser.add_argument('-e', '--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int, 
                        metavar='N',
                        help='number of batchsize')         

    ##################################################################################
    # 这里指的是当前进程在当前机器中的序号，注意和在全部进程中序号的区别，即指的是GPU序号0,1,2,3。
    # 在ENV模式中，这个参数是必须的，由启动脚本自动划分，不需要手动指定。要善用local_rank来分配GPU_ID。
    # 不需要填写，脚本自动划分
    parser.add_argument("--local_rank", type=int,                                    #
                        help='rank in current node')                                 #
    # 是否使用混合精度
    parser.add_argument('--use_mix_precision', default=False,                        #
                        action='store_true', help="whether to use mix precision")    #
    # Need 每台机器使用几个进程，即使用几个gpu  双卡2，
    parser.add_argument("--nproc_per_node", type=int,                                    #
                        help='numbers of gpus')                                 #
    # 分布式训练使用几台机器，设置默认1，单机多卡训练
    parser.add_argument("--nnodes", type=int, default=1, help='numbers of machines')
    # 分布式训练使用的当前机器序号，设置默认0，单机多卡训练只能设置为0
    parser.add_argument("--node_rank", type=int, default=0, help='rank of machines')
    # 分布式训练使用的0号机器的ip，单机多卡训练设置为默认本机ip
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                        help='ip address of machine 0')
    ##################################################################################                  
    args = parser.parse_args()
    #################################
    # train(args.local_rank, args)：一般情况下保持local_rank与进程所用GPU_ID一致。
    print("----------")
    print(args.local_rank)
    print(args.batch_size)
    print("------------")
    return args
    # exit()
        #
    #################################


if __name__ == '__main__':
    args = parse_args()
    train(args.local_rank, args)
