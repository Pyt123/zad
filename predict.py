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

global_counter = 0
TRAINING_COUNT = 7500
VALIDATE_COUNT = 500

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


def get_predicted(x):
    save_model_as_numpy(model)
    (w1, p1, w2, p2, w3, p3, w4, p4) = load_model_from_file()
    w1 = np.transpose(w1).copy(order='C')
    w2 = np.transpose(w2).copy(order='C')
    w3 = np.transpose(w3).copy(order='C')
    w4 = np.transpose(w4).copy(order='C')

    layer1 = np.empty((336,), order='C')
    layer2 = np.empty((336,), order='C')
    layer3 = np.empty((336,), order='C')
    layer4 = np.empty((36,), order='C')

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


EPOCHS = 50000
EPOCHS_TO_CHANGE = 400


def train(num_of_try, learning_rate, epsilon):
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
    # Load model

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
            NEXT_TO_CHANGE += EPOCHS_TO_CHANGE
            permute_train_set()
            inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
            targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
            targets = targets.squeeze(1)

            ratio = test(y_val)
            torch.save(model, 'notend' + str(num_of_try) + '.pth')
            if ratio > LAST_BEST:
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


def test(y_val):
    good = 0
    pred = get_predicted(x_val)
    #model.eval()
    for i in range(len(y_val)):
        #pred = model(x_val[i]).cpu().data.numpy()
        #if pred.argmax() == y_val[i]:
        if pred[i] == y_val[i]:
            good += 1
        #else:
         #   print(str(int(pred[i])) + '\t' + str(int(y_val[i])))
    ratio = (good / len(y_val)) * 100
    print("ratio: " + str(ratio))
    return ratio


def permute_train_set():
    global x, y, x_val, y_val, x_train, y_train
    arr = []
    for i in range(len(x)):
        arr.append((x[i], y[i]))

    arr = np.random.permutation(arr)
    newX = []
    newY = []
    for i in range(len(x)):
        (xp, yp) = arr[i]
        newX.append(xp)
        newY.append(yp)

    (x_train, y_train) = (np.asarray(newX[:TRAINING_COUNT]), np.asarray(newY[:TRAINING_COUNT]))
    (x_val, y_val) = (np.asarray(newX[(30164-500):]), np.asarray(newY[(30164-500):]))


print("start")

(x, y) = pkl.load(open('train.pkl', mode='rb'))
(x_val, y_val) = (x[(30164-500):], y[(30164-500):])
(x_train, y_train) = (x[:TRAINING_COUNT], y[:TRAINING_COUNT])

permute_train_set()
model = torch.load('mytraining.pth')

(x_val, y_val) = (x[TRAINING_COUNT:], y[TRAINING_COUNT:])
#x_val = torch.autograd.Variable(torch.from_numpy(x_val).type(torch.cuda.FloatTensor), requires_grad=True)
#targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
INCREASE_EPOCHS = 1000
learning_rates = [0.000005]
epsilons = [0.001]
for i in range(len(learning_rates)):
    for j in range(len(epsilons)):
        print('Testing learning rate = ' + str(learning_rates[i]) + ' and epsilon = ' + str(epsilons[j]))
        train(i + 100 * j, learning_rates[i], epsilons[j])
        print('\n')
    print('\n')

#model = torch.load('mytraining.pth')
#save_model_as_numpy(model)
#load_model_from_file()
#print("now")
#get_predicted(x_train)
#print('saved as numpy')
exit(0)

#save_model_as_numpy(model)
#print('saved as numpy')
