import numpy as np
from random import random

class PHEStruct:
    def __init__(self, num_arm, a):
        self.d = num_arm
        self.a = a
        # self.UserArmMean = np.zeros(self.d)
        # self.UserArmTrials = np.zeros(self.d)
        self.UserArmConfidence = np.zeros(self.d)
        self.UserArmHistory = [[] for _ in range(self.d)]
        self.UserArmTheta = np.zeros(self.d)

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        # self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        # self.UserArmTrials[articlePicked_id] += 1
        self.UserArmHistory[articlePicked_id].append(click)

        self.time += 1

    def getTheta(self):
        return self.UserArmTheta

    def decide(self, pool_articles):
        # self.updateConfidences(pool_articles)
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            n = len(self.UserArmHistory[article.id])
            self.UserArmConfidence[article.id] = (np.sqrt(2 * np.log(self.time) / n) 
                                                    if n != 0 else float('inf'))
            self.UserArmTheta[article.id] = ((sum(self.UserArmHistory[article.id]) + sum([1 for _ in range(self.a * n) if random() < 0.5])) / (n * (1 + self.a))
                                                    if n != 0 else float('inf'))
            article_pta = self.UserArmTheta[article.id] + self.UserArmConfidence[article.id]
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class PHEMultiArmedBandit:
    def __init__(self, num_arm, a=4):
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


