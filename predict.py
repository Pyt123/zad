# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import content
import time
import torch
import pickle as pkl
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from google.colab import files

global_counter = 0
VALIDATE_COUNT = 0
TRAINING_COUNT = 30134 - VALIDATE_COUNT

def save_model_as_numpy(model):
    i = 1
    for parameter in model.parameters():
        nump = parameter.cpu().type(torch.FloatTensor).data.numpy().astype(np.float16)
        np.save('model/params' + str(i), nump, allow_pickle=True)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'),
            np.load('model/params3.npy'), np.load('model/params4.npy'),
            np.load('model/params5.npy'), np.load('model/params6.npy'),
            np.load('model/params7.npy'), np.load('model/params8.npy'))


def save_model_as_txt(par1, par2, par3, par4):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)


def relu(x):
    return x * (x > 0)


def get_predicted(x, model):
    save_model_as_numpy(model)
    (w1, p1, w2, p2, w3, p3, w4, p4) = load_model_from_file()
    w1 = np.transpose(w1).copy(order='C')
    w2 = np.transpose(w2).copy(order='C')
    w3 = np.transpose(w3).copy(order='C')
    w4 = np.transpose(w4).copy(order='C')

    layer1 = np.empty((336,), order='C')#, dtype=np.float32)
    layer2 = np.empty((336,), order='C')#, dtype=np.float32)
    layer3 = np.empty((336,), order='C')#, dtype=np.float32)
    layer4 = np.empty((36,), order='C')#, dtype=np.float32)

    output_array = []
    length = len(x)
    for i in range(0, length):
        np.matmul(x[i], w1, out=layer1)
        layer1 += p1
        layer1 = relu(layer1)
        
        np.matmul(layer1, w2, out=layer2)
        layer2 += p2
        layer2 = relu(layer2)
        
        np.matmul(layer2, w3, out=layer3)
        layer3 += p3
        layer3 = relu(layer3)
        
        np.matmul(layer3, w4, out=layer4)
        layer4 += p4
        layer4 = relu(layer4)
        output_array.append(layer4.argmax())

    output_vector = np.array(output_array)
    return output_vector.reshape((len(output_vector), 1))


EPOCHS = 200
EPOCHS_TO_CHANGE = 5000


def train(x_train, y_train, num_of_try, learning_rate, epsilon):
    #START_MOMENTUM = 1
    #MOMENTUM = START_MOMENTUM
    #START_LR = 1
    #LR = START_LR
    #DIVIDER_MOM = 1.03
    #DIVIDER_LR = 1.05
    NEXT_TO_CHANGE = EPOCHS_TO_CHANGE
    LAST_BEST = 0
    BEST_EPOCH = 0

    # Create model
    #model = content.NeuralNet().cuda()
    # Load model
    model = torch.load('torchmodel' + str(num_of_try) + '.pth')
    print("Loaded...")
    # Some stuff
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Convert numpy arrays to torch variables
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
    targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
    targets = targets.squeeze(1)

    for epoch in range(EPOCHS):
        if epoch == NEXT_TO_CHANGE:
            #MOMENTUM /= DIVIDER_MOM
            #LR /= DIVIDER_LR
            #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
            NEXT_TO_CHANGE += EPOCHS_TO_CHANGE
            ratio = test(model, y_val)
            #if ratio > 88:
                #optimizer = optim.Adam(model.parameters(), lr=learning_rate/10, eps=epsilon)
            #elif ratio > 85:
                #optimizer = optim.Adam(model.parameters(), lr=learning_rate/2.5, eps=epsilon)
            #elif ratio > 87:
                #optimizer = optim.Adam(model.parameters(), lr=learning_rate/5, eps=epsilon)


            if ratio > LAST_BEST:
                torch.save(model, 'bestmodel' + str(num_of_try) + '.pth')
                LAST_BEST = ratio
                BEST_EPOCH = epoch
                print('\n-----Best epoch: ' + str(BEST_EPOCH) + '\tBest ratio: ' + str(LAST_BEST) + '-----\n')

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % EPOCHS_TO_CHANGE == 0:
            print('Epoch [{}/{}],\tLoss: {:.24f}'.format(epoch, EPOCHS, loss.data[0]))

    torch.save(model, 'endingmodel' + str(num_of_try) + '.pth')
    print('END-----Best epoch: ' + str(BEST_EPOCH) + '\tBest ratio: ' + str(LAST_BEST) + '-----\n')
    return model


def test(model, y_val):
    good = 0
    pred = get_predicted(x_val, model)
    model.eval()
    for i in range(0, VALIDATE_COUNT):
        #pred = model(x_val[i]).cpu().data.numpy()
        #if pred.argmax() == y_val[i]:
        if pred[i] == y_val[i]:
            good += 1
        #else:
         #   print(str(int(pred[i])) + '\t' + str(int(y_val[i])))
    ratio = (good / VALIDATE_COUNT) * 100
    model.train()
    print("ratio: " + str(ratio))
    return ratio


(x, y) = pkl.load(open('train.pkl', mode='rb'))
(x_train, y_train) = (x[VALIDATE_COUNT:TRAINING_COUNT], y[VALIDATE_COUNT:TRAINING_COUNT])
(x_val, y_val) = (x[:VALIDATE_COUNT], y[:TRAINING_COUNT])
#x_val = torch.autograd.Variable(torch.from_numpy(x_val).type(torch.cuda.FloatTensor), requires_grad=True)
#targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
INCREASE_EPOCHS = 1000
learning_rates = [0.0005]
epsilons = [0.001]
for i in range(len(learning_rates)):
    for j in range(len(epsilons)):
        print('Testing learning rate = ' + str(learning_rates[i]) + ' and epsilon = ' + str(epsilons[j]))
        train(x_train, y_train, i + 100 * j, learning_rates[i], epsilons[j])
        print('\n')
    print('\n')
    EPOCHS += INCREASE_EPOCHS

#model = torch.load('bestmodel0.pth')
#save_model_as_numpy(model)
#load_model_from_file()
#test(model, y_val)
#print('saved as numpy')
#from google.colab import files
#files.download('bestmodel0.pth')
#files.download('bestmodel1.pth')
#files.download('bestmodel2.pth')
exit(0)

#save_model_as_numpy(model)
#print('saved as numpy')
