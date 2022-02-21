import numpy as np
import scipy.stats as sps
import random
import matplotlib.pyplot as plt


class MathSampleParameters:

    def __init__(self, sampleSize):
        self.size = sampleSize

    def generateNormalSample(self, local, var):
        return [np.random.normal(loc=local, scale=var) for _ in range(self.size)]

    def generateUniformSample(self):
        return [np.random.uniform(low=0, high=100) for _ in range(self.size)]

    def generateRayleighSample(self):
        return [np.random.rayleigh() for _ in range(self.size)]

    def generateExponentialSample(self):
        return [np.random.exponential() for _ in range(self.size)]

    def generateGammaSample(self):
        return [np.random.gamma(2.) for _ in range(self.size)]

    def generateWeibullSample(self):
        return [np.random.weibull(5.) for _ in range(self.size)]

    def generateLognormalSample(self):
        return [np.random.lognormal() for _ in range(self.size)]

    def generateBetaSample(self):
        return [np.random.beta(0, 1) for _ in range(self.size)]

    @staticmethod
    def expectation(array):
        return np.mean(array)

    @staticmethod
    def variation(array):
        return np.var(array)

    @staticmethod
    def skewness(array):
        return sps.skew(array)

    @staticmethod
    def median(array):
        return np.median(array)

    @staticmethod
    def mode(array):
        return float(sps.mode(array)[0])


def main():
    m = MathSampleParameters(5)
    sample = m.generateNormalSample(2.5,1)
    print(sample)

    x_value = [value for value in range(0, 5, 1)]
    plt.hist(sample, x_value)
    plt.axvline(x=m.median(sample), color='r')
    plt.axvline(x=m.mode(sample), color='m')
    plt.axvline(x=m.expectation(sample), color='y')
    plt.show()

    print(f"M = {m.expectation(sample)}")
    print(f"D = {m.variation(sample)}")
    print(f"SK = {m.skewness(sample)}")
    print(f"Mode = {m.mode(sample)}")
    print(f"Median = {m.median(sample)}")


if __name__ == '__main__':
    main()
