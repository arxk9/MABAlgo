import numpy as np
from random import random
from math import ceil

class PHEStruct:
    def __init__(self, num_arm, a):
        self.d = num_arm
        self.a = a
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d, dtype=int)

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1

        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            n = self.UserArmTrials[article.id]
            if n == 0:
                return article
            article_pta = (self.UserArmMean[article.id] * n + sum([1 for _ in range(ceil(self.a * n)) if random() < 0.5])) / ceil(n * (1 + self.a))
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class PHEMultiArmedBandit:
    def __init__(self, num_arm, a=0.5):
        self.users = {}
        self.num_arm = num_arm
        self.a = a
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm, self.a)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean


