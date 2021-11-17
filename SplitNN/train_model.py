import torch
from torch import nn, optim

from utils import device, trainloader, testloader
from dla_simple import SimpleDLA

net = SimpleDLA()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f' accuracy e ceva {100.*correct/total} si loss e {train_loss/(batch_idx+1)}')


def test(epoch):
    best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f' accuracy e ceva {100.*correct/total} si loss e {test_loss/(batch_idx+1)}')


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        print(f'Acc is {acc} and best acc so far was {best_acc}')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        PATH = "model.pt"

        torch.save(net.state_dict(), PATH)
        best_acc = acc


# we had 100 epochs
for epoch in range(5):
    train(epoch)
    test(epoch)
    scheduler.step()

