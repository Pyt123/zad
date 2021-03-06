# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import content
import torch
import pickle as pkl
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

global_counter = 0
TRAINING_COUNT = 28134
VALIDATE_COUNT = 2000


def save_model_as_numpy(model):
    i = 1
    for parameter in model.parameters():
        nump = parameter.cpu().type(torch.FloatTensor).data.numpy().astype(np.float16)
        if i % 2 == 1:
            nump = np.transpose(nump)
        np.save('model/params' + str(i), nump, allow_pickle=True)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'),
            np.load('model/params3.npy'), np.load('model/params4.npy'),
            np.load('model/params5.npy'), np.load('model/params6.npy'),
            np.load('model/params7.npy'), np.load('model/params8.npy'),
            np.load('model/params9.npy'), np.load('model/params10.npy'),
            np.load('model/params11.npy'), np.load('model/params12.npy'))


def save_model_as_txt(par1, par2, par3, par4):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)


def relu(x):
    return x * (x > 0)


def get_predicted(x):
    save_model_as_numpy(model)
    (w1, p1, w2, p2, w3, p3, w4, p4, w5, p5, w6, p6) = load_model_from_file()
    '''w1 = np.transpose(w1)
    w2 = np.transpose(w2)
    w3 = np.transpose(w3)
    w4 = np.transpose(w4)'''

    layer1 = np.empty((336,), order='C')
    layer2 = np.empty((336,), order='C')
    layer3 = np.empty((336,), order='C')
    layer4 = np.empty((336,), order='C')
    layer5 = np.empty((336,), order='C')
    layer6 = np.empty((36,), order='C')

    output_array = []
    length = len(x)
    for i in range(0, length):
        np.matmul(x[i], w1, out=layer1)
        lay1 = relu(layer1 + p1)

        np.matmul(lay1, w2, out=layer2)
        lay2 = relu(layer2 + p2)

        np.matmul(lay2, w3, out=layer3)
        lay3 = relu(layer3 + p3)

        np.matmul(lay3, w4, out=layer4)
        lay4 = relu(layer4 + p4)

        np.matmul(lay4, w5, out=layer5)
        lay5 = relu(layer5 + p5)

        np.matmul(lay5, w6, out=layer6)
        lay6 = relu(layer6 + p6)

        output_array.append(lay6.argmax())

    output_vector = np.array(output_array)
    return output_vector.reshape(length, 1)


EPOCHS = 10000
EPOCHS_TO_CHANGE = 300


def train(num_of_try, learning_rate, epsilon):
    NEXT_TO_CHANGE = EPOCHS_TO_CHANGE
    LAST_BEST = 0
    BEST_EPOCH = 0

    # Some stuff
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Convert numpy arrays to torch variables
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
    targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
    targets = targets.squeeze(1)

    for epoch in range(EPOCHS):
        if epoch == NEXT_TO_CHANGE:
            NEXT_TO_CHANGE += EPOCHS_TO_CHANGE
            #permute_train_set()
            #inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
            learning_rate /= 1.1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
            print('lr = ' + str(learning_rate))

            ratio = test()
            torch.save(model, 'notend' + str(num_of_try) + '.pth')
            if ratio > LAST_BEST:
                LAST_BEST = ratio
                BEST_EPOCH = epoch
                torch.save(model, 'bestmodel' + str(num_of_try) + '.pth')
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


def test():
    good = 0
    pred = get_predicted(x_val)
    for i in range(len(y_val)):
        if pred[i] == y_val[i]:
            good += 1

    ratio = (good / len(y_val)) * 100
    print("ratio: " + str(ratio))
    return ratio


def permute_train_set():
    global x, y, x_val, y_val, x_train, y_train
    arr = []
    for i in range(len(x) - VALIDATE_COUNT):
        arr.append((x[i], y[i]))

    arr = np.random.permutation(arr)
    newX = []
    newY = []
    for i in range(len(x) - VALIDATE_COUNT):
        (xp, yp) = arr[i]
        newX.append(xp)
        newY.append(yp)

    (x_train, y_train) = (np.asarray(newX[:TRAINING_COUNT]), np.asarray(newY[:TRAINING_COUNT]))


(x, y) = pkl.load(open('train.pkl', mode='rb'))
(x_val, y_val) = (x[(30134-VALIDATE_COUNT):], y[(30134-VALIDATE_COUNT):])
(x_train, y_train) = (x[:TRAINING_COUNT], y[:TRAINING_COUNT])

#model = torch.load('mytraining.pth')
model = content.NeuralNet().cuda()

learning_rates = [0.025]
epsilons = [0.001]
for i in range(len(learning_rates)):
    for j in range(len(epsilons)):
        print('Testing learning rate = ' + str(learning_rates[i]) + ' and epsilon = ' + str(epsilons[j]))
        train(i + 100 * j, learning_rates[i], epsilons[j])
        print('\n')
    print('\n')
'''
#model = torch.load('mytraining.pth')
#save_model_as_numpy(model)
#load_model_from_file()
#print("now")
#save_model_as_numpy(model)
#get_predicted(x_train)
#print('saved as numpy')
exit(0)

#save_model_as_numpy(model)
#print('saved as numpy')
'''
exit(0)
