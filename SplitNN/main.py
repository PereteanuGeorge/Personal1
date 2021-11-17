import torch

from client import model1
from utils import load_input, context
import tenseal as ts
from utils import device, testloader
from server import enc_model, model2, model3, enc_model3
import time

criterion = torch.nn.CrossEntropyLoss()


def test():
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model1(inputs)

        if batch_idx % 500 == 0:
            print(f'we are at batch {batch_idx}')

        # Image encoding
        x_enc = [ts.ckks_vector(context, x.tolist()) for x in preds]

        enc_output = enc_model(x_enc)

        outputs = enc_output.decrypt()
        outputs = torch.tensor(outputs).view(1, -1)
        outputs = outputs.to(device)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    print(f' accuracy in test model is {100. * correct / total} and loss is {test_loss / (batch_idx + 1)}')



def test_splitted_models():
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model1(inputs)

            # Image encoding
            outputs = model2(preds)
            # poate trebuie si asta
            #         output = torch.tensor(output).view(1, -1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f' accuracy in test splitted model is {100. * correct / total} and loss is {test_loss / (batch_idx + 1)}')


if __name__ == "__main__":
    print("Starting main")
    image, target = load_input()
    print(f'Target goal is {target}')
    image = image.to(device)
    start = time.time()
    preds = model1(image)
    #
    # # Image encoding
    x_enc = [ts.ckks_vector(context, x.tolist()) for x in preds]
    #
    enc_output = enc_model(x_enc)
    #
    result = enc_output.decrypt()
    #
    probs = torch.softmax(torch.tensor(result), 0)
    label_max = torch.argmax(probs)
    print("Maximum probability for label {}".format(label_max))
    end = time.time()
    print(f'total time taken to do the inference {end - start}')

    #start = time.time()
    #test()
    #test_splitted_models()
    #end = time.time()
    #print(f'total time taken to do the inference on test set is {end - start}')

    pytorch_total_params1 = sum(p.numel() for p in model1.parameters())
    pytorch_total_params2 = sum(p.numel() for p in model2.parameters())
    print(f'total number of params is {pytorch_total_params1 + pytorch_total_params2}')
    #
    # start = time.time()
    # test()
    # test_splitted_models()
    # end = time.time()
    # print(f'total time taken to do the inference on test set is {end - start}')
