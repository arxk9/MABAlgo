import numpy as np
from random import random
from math import ceil

class LinPHEStruct:
    def __init__(self, featureDimension, lambda_, a):
        self.d = featureDimension
        self.A = (a + 1) * lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_
        self.a = a
        self.b = np.zeros(self.d)
        self.bsum = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.featureHistory = dict()
        self.seen = set()
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        featureString = str(articlePicked_FeatureVector)
        if featureString not in self.featureHistory:
            self.featureHistory[featureString] = [articlePicked_FeatureVector, 1]
        else:
            self.featureHistory[featureString][1] += 1

        self.A += (self.a + 1) * np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.bsum += articlePicked_FeatureVector * click
        self.b = self.bsum
        for thing in [self.featureHistory[s] for s in self.featureHistory]:
            perturbation = np.random.binomial(ceil(self.a * thing[1]), 0.5)
            self.b += thing[0] * perturbation
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = self.AInv @ self.b
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            if article not in self.seen:
                self.seen.add(article)
                return article
            article_pta = np.dot(self.UserTheta, article.featureVector)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class LinPHEBandit:
    def __init__(self, dimension, lambda_, a=0.5):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.a = a
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinPHEStruct(self.dimension, self.lambda_, self.a)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


