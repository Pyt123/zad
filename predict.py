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
import numpy as np


def save_model_as_numpy(model):
    i = 1
    for parameter in model.parameters():
        nump = parameter.cpu().type(torch.FloatTensor).data.numpy().astype(np.float16)
        np.save('model/params' + str(i), nump, allow_pickle=True)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'),
            np.load('model/params3.npy'), np.load('model/params4.npy'))


def save_model_as_txt(par1, par2, par3, par4):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)


def relu(x):
    return x * (x > 0)


def get_predicted(x):
    (w1, _, w2, _) = load_model_from_file()
    w1 = np.transpose(w1).copy(order='C')
    w2 = np.transpose(w2).copy(order='C')
    output_array = []
    layer1 = np.empty((900,), order='C')
    layer2 = np.empty((36,), order='C')
    length = len(x)
    for i in range(0, length):
        np.matmul(x[i], w1, out=layer1)
        layer1 = relu(layer1)
        np.matmul(layer1, w2, out=layer2)
        layer2 = relu(layer2)
        output_array.append(layer2.argmax())

    output_vector = np.array(output_array)
    return output_vector.reshape((len(output_vector), 1))


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """

    # -------------------------------------------------------------
    print('start')
    start_time = time.time()
    (w1, _, w2, _) = load_model_from_file()
    w1 = np.transpose(w1)
    w2 = np.transpose(w2)
    w1 = w1.copy(order='C')
    w2 = w2.copy(order='C')
    output_array = []
    length = len(x)
    #output = np.empty((length,), order='C')
    layer1 = np.empty((900,), order='C')
    layer2 = np.empty((36,), order='C')
    #layer1 = np.asarray(layer1)
    #layer2 = np.asarray(layer2)
    for i in range(0, length):
        np.matmul(x[i], w1, out=layer1)
        layer1 = relu(layer1)
        np.matmul(layer1, w2, out=layer2)
        layer2 = relu(layer2)
        #output.setfield(np.int(arg), np.int, i)
        output_array.append(layer2.argmax())

    output = np.array(output_array)
    print('time needed: ' + str(time.time() - start_time) + ' s')
    #return output
    # -------------------------------------------------------------
    np.save('predicted_vals', output)

    good = 0
    for i in range(0, 2634):
        if output[i] == y_train[i]:
            good += 1
    print("good: " + str(good))
    print("ratio: " + str(good/2634.0 * 100))
    return output


content.train()

(x_train, y_train) = pkl.load(open('train.pkl', mode='rb'))
(x_train, y_train) = (x_train[27500:], y_train[27500:])

model = torch.load('mytraining.pth', 'cpu')
save_model_as_numpy(model)
print('saved as numpy')

pred = get_predicted(x_train)

good = 0
for i in range(0, 2634):
    if pred[i] == y_train[i]:
        good += 1
print("good: " + str(good))
print("ratio: " + str(good / 2634.0 * 100))

exit(0)
