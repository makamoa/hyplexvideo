import numpy as np
from scipy.integrate import trapz as integr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from sklearn.linear_model import LinearRegression, Lasso
from scipy.interpolate import interp1d
from scipy.optimize import nnls

class Sample:
    def __init__(self, wl, X, Y):
        self.wl = wl
        self.X = X
        self.Y = Y


class NNLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if len(y.shape) > 1:
            nchannels = y.shape[1]
        else:
            nchannels = 1
            y = y.reshape(-1, 1)
        self.nchannels = nchannels
        ### add 0 dimension
        XX = np.ones([X.shape[0], X.shape[1] + 1])
        XX[:, 1:] = X[:, :]
        if not self.fit_intercept:
            XX = X.copy()
        a = np.ones([XX.shape[1], y.shape[1]])
        for i in range(nchannels):
            a[:, i], _ = nnls(XX, y[:, i])
        if self.fit_intercept:
            self.coef_ = a[1:, :].T.copy()
            self.intercept_ = a[0, :]
        else:
            self.intercept_ = np.zeros(nchannels)
            self.coef_ = a.T.copy()

    def predict(self, X):
        return X @ self.coef_.T + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        return MSE(y, y_pred)


class MetaCalibration():
    def __init__(self,samples,nmodes=5,fitting='sin',regression=LinearRegression):
        nsamples = len(samples)
        """ 
        X - coefficient matrix <Pi,CMFr/g/b>, 
        with sizes [nsamples,nmodes]
        y - output vector with sizes [nsamples,3] 3 is for XYZ
        """
        if fitting=='sin':
            self.t_n=self.sin_n
        elif fitting=='poly':
            self.t_n=self.poly_n
        elif fitting == 'gaussian':
            self.t_n = self.gaussian
        elif fitting == 'lorenzian':
            self.t_n = self.gaussian
        else:
            ValueError('Unrecognized function type!')
        self.nmodes=nmodes
        self.nfeatures = samples[0].Y.__len__()
        X = np.zeros([nsamples,nmodes])
        y = np.zeros([nsamples,self.nfeatures])
        self.wl =samples[0].wl
        self.N = len(self.wl)
        for i,sample in enumerate(samples):
            P = sample.X
            Y = sample.Y
            #W [1,nmodes]
            #X[i,:]=[integr(P*self.t_n(j),self.wl) for j in range(nmodes)]
            X[i, :] = [P @ self.t_n(j) for j in range(nmodes)]
            # same for output vector y
            y[i,:] = Y[:]
        self.X = X
        self.y = y
        self.model = regression(fit_intercept=False)
        self.model.fit(X,y)
        self.a = self.model.coef_
        self.CMFs = self.get_CMF_XYZ()

    @classmethod
    def load(cls,fname='camera.npy'):
        with open(fname,'rb') as file:
            camera = pickle.load(file)
        return camera

    def get_CMF_XYZ(self):
        XYZ=[]
        for coeff in self.a:
            res = np.zeros(len(self.wl))
            for j, k in enumerate(coeff):
                res += k * self.t_n(j)
            f=interp1d(self.wl,res)
            XYZ.append(f)
        return XYZ

    def spectra_to_features(self,wl,spectra):
        Y=[]
        for CMF in self.CMFs:
            #Y.append(integr(spectra*CMF(wl), wl))
            Y.append(spectra @ CMF(wl))
        return np.array(Y) + self.model.intercept_

    def show(self,show=True):
        for i,CMF in enumerate(self.CMFs):
            plt.plot(self.wl, CMF(self.wl),label='$X_{%d\lambda}$' % i)
        plt.legend()
        plt.title('nmodes=%d' % self.nmodes)
        plt.xlabel('wl, nm')
        #plt.yticks([])
        if show:
            plt.show()

    def show_mode(self,n):
        coeff = self.a[n]
        positive = np.zeros(len(self.wl))
        negative = np.zeros(len(self.wl))
        for j, k in enumerate(coeff):
            if k>=0:
                positive += k * self.t_n(j)
            else:
                negative += k * self.t_n(j)
        plt.figure()
        plt.plot(self.wl,positive,label='positive')
        plt.plot(self.wl,-negative,label='negative')
        plt.plot(self.wl, positive+negative, label='result')
        plt.legend()
        plt.show()


    def score(self,samples):
        y_pred = []
        y_true = []
        for sample in samples:
            RGB_true = sample.Y
            RGB_pred = self.spectra_to_features(sample.wl, sample.X)
            y_pred.append(RGB_pred)
            y_true.append(RGB_true)
        y_true=np.array(y_true)
        y_pred = np.array(y_pred)
        return MSE(y_true,y_pred)

    def sin_n(self,n):
        x = np.arange(self.N)
        return 2 * np.sin(np.pi * (n + 1) * (x + 1) / (len(x) + 1)) / 2 / (self.N + 1)

    def poly_n(self,n):
        pass

    def gaussian(self,n,sigma=40):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        return np.exp(-np.abs(x-xi)**2/sigma**2)

    def lorenzian(self,n,sigma=20):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0,N,N/self.nmodes)[n]
        x = np.arange(self.N)
        y = (x-xi)/sigma/2
        return 1/(1+y**2)

    def save_modes(self):
        CMFs = np.stack([CMF(self.wl) for CMF in self.CMFs],axis=0)
        np.save('../data/CMFs-%d-gaussian.npy' % self.nmodes,CMFs)

    def get_modes(self):
        return np.stack([CMF(self.wl) for CMF in self.CMFs],axis=0)

    def save(self,fname='camera.npy'):
        with open(fname,'wb') as file:
            pickle.dump(self,file)


class FuncFitting():
    def __init__(self, wl, Y, nmodes=5, normalize=False, fitting='gaussian', fitting_params={},
                 regression=LinearRegression()):
        self.fitting_params = fitting_params
        if fitting == 'sin':
            self.t_n = self.sin_n
        elif fitting == 'poly':
            self.t_n = self.poly_n
        elif fitting == 'gaussian':
            self.t_n = self.gaussian
        elif fitting == 'lorenzian':
            self.t_n = self.lorenzian
        elif fitting == 'data':
            self.data = None
            self.t_n = self.from_data
        else:
            ValueError('Unrecognized function type!')
        self.wl = wl
        self.N = len(wl)
        self.nmodes = nmodes
        nsamples = Y.shape[0]
        X = np.zeros([self.N, nmodes])
        for i in range(nmodes):
            X[:, i] = self.t_n(i, **self.fitting_params)
        self.X = X
        self.y = Y.T
        self.model = regression
        self.model.fit(X, Y.T)
        self.score_ = self.model.score(X, Y.T)
        self.a = self.model.coef_
        self.fs = self.get_fs()

    def get_fs(self):
        fs = []
        for coeff in self.a:
            res = np.zeros(len(self.wl))
            for j, k in enumerate(coeff):
                res += k * self.t_n(j, **self.fitting_params)
            f = interp1d(self.wl, res)
            fs.append(f)
        return fs

    def get_compressed_from_spectra(self, spectra, intercept):
        res = np.zeros(self.nmodes)
        for i, coeff in enumerate(self.a):
            res[i] = coeff @ self.spectra_to_features(spectra, intercept)
        return res + self.model.intercept_

    def get_compressed(self, barcode):
        res = np.zeros(self.nmodes)
        for i, coeff in enumerate(self.a):
            for j, k in enumerate(coeff):
                res[i] += k * barcode[j]
        return res - self.model.intercept_

    def sin_n(self, n):
        x = np.arange(self.N)
        return 2 * np.sin(np.pi * (n + 1) * (x + 1) / (len(x) + 1)) / 2 / (self.N + 1)

    def poly_n(self, n):
        pass

    def spectra_to_features(self, spectra, intercept):
        Y=[]
        for i in range(self.nmodes):
            #Y.append(integr(spectra*CMF(wl), wl))
            Y.append(spectra @ self.t_n(i, **self.fitting_params))
        return np.array(Y) + intercept

    def gaussian(self, n, sigma=150):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0, N, N / self.nmodes)[n]
        x = np.arange(self.N)
        return np.exp(-np.abs(x - xi) ** 2 / sigma ** 2)

    def lorenzian(self, n, sigma=10):
        N = self.N + (self.N % self.nmodes)
        xi = np.arange(0, N, N / self.nmodes)[n]
        x = np.arange(self.N)
        y = (x - xi) / sigma / 2
        return 1 / (1 + y ** 2)

    def from_data(self, n, arr, transform=(lambda x: x)):
        if self.data is None:
            self.data = arr
        f = transform(self.data[n])
        return f


if __name__ == '__main__':
    samples = np.load('../data/samples.npy', allow_pickle=True)
    camera=MetaCalibration(samples, fitting='gaussian', nmodes=8)
    score = camera.score(samples)
    camera.show()
    print(score)