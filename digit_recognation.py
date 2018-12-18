import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import sys

'''funkcja wczytania danych MNIST'''


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist('mnist', kind='train')
print('Rzędy: %d, Kolumny: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rzędy: %d, kolumny: %d' % (X_test.shape[0], X_test.shape[1]))

print(X_train)
print(X_train[41])
print(X_train[41].reshape(28, 28))

print(y_train)
print(y_train[41])

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(2):
    img = X_train[41].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(5):
    img = X_train[y_train == 0][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

'''Wyświetlenie po jednym obrazie każdej klasy'''

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

'''Wyświetlenie 25 odmian jednej klasy'''

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):
    img = X_train[y_train == 8][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


class NeuralNetMLP(object):
    """ Sieć neuronowa z dodatnim sprzężeniem zwrotnym / klasyfikator - wielowarstwowy perceptron.

    Parametry
    ------------
    n_output : liczba całkowita
      Liczba jednostek wyjściowych, powinna być taka sama
      jak liczba unikatowych etykiet klas.

    n_features : liczba całkowita
      Liczba cech (wymiarów) w docelowym zestawie danych.
      Powinna być równa liczbie kolumn w tablicy X.

    n_hidden : liczba całkowita (domyślnie: 30)
      Liczba jednostek ukrytych.

    l1 : wartość zmiennoprzecinkowa (domyślnie: 0.0)
      Parametr lambda regularyzacji L1.
      Nie ma regularyzacji, jeśli l1=0.0 (domyślnie)

    l2 : wartość zmiennoprzecinkowa (domyślnie: 0.0)
      Parametr lambda regularyzacji L2.
      Nie ma regularyzacji, jeśli l2=0.0 (domyślnie)

    epochs : liczba całkowita (domyślnie: 500)
      Liczba przebiegów po zestawie danych uczących.

    eta : wartość zmiennoprzecinkowa (domyślnie: 0.001)
      Współczynnik uczenia.

    alpha : wartość zmiennoprzecinkowa (domyślnie: 0.0)
      Stała momentu. Czynnik przemnażany przez gradient
      poprzedniej epoki t-1 poprawiający szybkość uczenia
      w(t) := w(t) - (grad(t) + alpha*grad(t-1))

    decrease_const : wartość zmiennoprzecinkowa (domyślnie: 0.0)
      Stała redukująca. Zmniejsza wartość współczynnika uczenia
      po każdej epoce za pomocą wzoru eta / (1 + epoch*decrease_const)

    shuffle : typ boolowski (domyślnie: True)
      Jeżeli wartość jest równa "True", tasuje dane uczące po każdej epoce
      w celu uniknięcia cykliczności.

    minibatches : liczba całkowita (domyślnie: 1)
      Dzieli dane uczące na k podzbiorów w celu zwiększenia wydajności.
      Optymalizacja metodą gradientu prostego, jeśli k=1 (domyślnie).

    random_state : liczba całkowita (domyślnie: None)
      Wprowadza losowość podczas tasowania i inicjowania wag.

    Atrybuty
    -----------
    cost_ : lista
      Suma kwadratów błędów po każdej epoce.

    """

    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Koduje etykiety do postaci "gorącojedynkowej"

        Parametry
        ------------
        y : tablica, postać = [n_próbek]
            Wartości docelowe.

        Zwraca
        -----------
        onehot : tablica, postać = (n_etykiet, n_próbek)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Inicjuje wagi mające niewielkie, losowe wartości."""
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Oblicza funkcję logistyczną (sigmoidalną).

        Wykorzystuje funkcję scipy.special.expit w celu uniknięcia
        błędu przepełnienia bufora (ang. overflow error) dla bardzo małych
        wartości parametru z.

        """
        # zwraca 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Oblicza gradient funkcji logistycznej"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Dodaje do tablicy jednostkę obciążenia (kolumnę lub rząd jedynek) o indeksie 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('Atrybut `how` musi mieć wartość `column` lub `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Oblicza krok sprzężenia w przód

        Parametry
        -----------
        X : tablica, postać = [n_próbek, n_cech]
          Warstwa wejściowa zawierająca pierwotne cechy.

        w1 : tablica, postać = [n_jednostek_ukrytych, n_cech]
          Macierz wag łączących warstwę wejściową z warstwą ukrytą.

        w2 : tablica, postać = [n_jednostek_wyjściowych, n_jednostek_ukrytych]
          Macierz wag łączących warstwę ukrytą z warstwą wyjściową.

        Zwraca
        ----------
        a1 : tablica, postać = [n_próbek, n_cech+1]
          Wartości wejściowe wraz z jednostką obciążenia.

        z2 : tablica, postać = [n_ukryta, n_próbek]
          Całkowita wartość pobudzenia warstwy ukrytej.

        a2 : tablica, postać = [n_ukryta+1, n_próbek]
          Aktywacja warstwy ukrytej.

        z3 : tablica, postać = [n_jednostek_wyjściowych, n_próbek]
          Całkowwita wartość pobudzenia warstwy wyjściowej.

        a3 : tablica, postać = [n_jednostek_wyjściowych, n_próbek]
          Aktywacja warstwy wyjściowej.

        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Oblicza koszt regularyzacji L2"""
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Oblicza koszt regularyzacji L1"""
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """Oblicza funkcję kosztu.

        y_enc : tablica, postać = (n_etykiet, n_próbek)
          Etykiety klas zakodowane do postaci "gorącojedynkowej".

        output : tablica, postać = [n_jednostek_wyjściowych, n_próbek]
          Aktywacja warstwy wyjściowej (sprzężenie w przód).

        w1 : tablica, postać = [n_jednostek_ukrytych, n_cech]
          Macierz wag łączących warstwę wejściową z warstwą ukrytą.

        w2 : tablica, postać = [n_jednostek wyjściowych, n_jednostek_ukrytych]
          Macierz wag łączących warstwę ukrytą z warstwą wyjściową.

        Zwraca
        ---------
        cost : wartość zmiennoprzecinkowa
          Regularyzowana funkcja kosztu.

         """
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Oblicza krok gradientu za pomocą wstecznej propagacji.

        Parametry
        ------------
        a1 : tablica, postać = [n_próbek, n_cech+1]
          Jednostki wejściowe wraz z jednostką obciążenia.

        a2 : tablica, postać = [n_ukryta+1, n_próbek]
          Aktywacja warstwy ukrytej.

        a3 : tablica, postać = [n_jednostek_wyjściowych, n_próbek]
          Aktywacja warstwy wyjściowej.

        z2 : tablica, postać = [n_ukryta, n_próbek]
          Całkowita wartość pobudzenia warstwy ukrytej.

        y_enc : tablica, postać = (n_etykiet, n_próbek)
          Etykiety klas zakodowane do postaci "gorącojedynkowej".

        w1 : tablica, postać = [n_jednostek_ukrytych, n_cech]
          Macierz wag łączących warstwę wejściową z warstwą ukrytą.

        w2 : tablica, postać = [n_jednostek_wyjściowych, n_jednostek_ukrytych]
          Macierz wag łączących warstwę ukrytą z warstwą wyjściową.

        Zwraca
        ---------

        grad1 : tablica, postać = [n_jednostek_ukrytych, n_cech]
          Gradient macierzy wag w1.

        grad2 : tablica, postać = [n_jednostek wyjściowych, n_jednostek_ukrytych]
            Gradient macierzy wag w2.

        """
        # propagacja wsteczna
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularyzacja
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def predict(self, X):
        """Prognozowanie etykiet klas

        Parametry
        -----------
        X : tablica, postać = [n_próbek, n_cech]
          Warstwa wejściowa z pierwotnymi cechami.

        Zwraca:
        ----------
        y_pred : tablica, postać = [n_próbek]
          Przewidywane etykiety klas.

        """
        if len(X.shape) != 2:
            raise AttributeError('X musi być macierzą [n_próbek, n_cech].\n'
                                 'Wprowadź X[:,None] dla klasyfikacji przy użyciu jednej cechy,'
                                 '\nlub X[[i]] dla klasyfikacji jednopróbkowej.')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Aktualizuje wagi za pomocą danych uczących.

        Parametry
        -----------
        X : tablica, postać = [n_próbek, n_cech]
          Warstwa wejściowa zawierająca pierwotne cechy.

        y : tablica, postać = [n_próbek]
          Docelowe etykiety klas.

        print_progress : typ boolowski (domyślnie: False)
          Wyświetla postępy jako stosunek liczby epok do
          standardowego strumienia błędów (stderr).

        Zwraca:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            # współczynnik uczenia adaptacyjnego
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoka: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # sprzężenie w przód
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)

                # oblicza gradient za pomocą wstecznej propagacji
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=100,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Koszt')
plt.xlabel('Epoki * 50')
plt.tight_layout()
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Koszt')
plt.xlabel('Epoki')
plt.tight_layout()
plt.show()


y_train_pred = nn.predict(X_train)

if sys.version_info < (3, 0):
    acc = (np.sum(y_train == y_train_pred, axis=0)).astype('float') / X_train.shape[0]
else:
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]

print('Dokładność wobec danych uczących: %.2f%%' % (acc * 100))


y_test_pred = nn.predict(X_test)

if sys.version_info < (3, 0):
    acc = (np.sum(y_test == y_test_pred, axis=0)).astype('float') / X_test.shape[0]
else:
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

print('Dokładność wobec danych testowych: %.2f%%' % (acc * 100))


miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) r: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()