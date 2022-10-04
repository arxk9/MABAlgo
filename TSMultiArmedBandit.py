import numpy as np

class TSStruct:
    def __init__(self, num_arm):
        self.d = num_arm
        sigma_0 = 5000
        self.UserArmPriors = [[0, sigma_0] for _ in range(self.d)]

        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.UserArmHistory = [[] for _ in range(self.d)]
        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1

        self.UserArmHistory[articlePicked_id].append(click) # may change this
        
        variance = (1 / self.UserArmPriors[articlePicked_id][1] ** -2 + self.UserArmTrials[articlePicked_id]) ** -1
        self.UserArmPriors[articlePicked_id][0] = sum(self.UserArmHistory[articlePicked_id]) / self.UserArmTrials[articlePicked_id]
        self.UserArmPriors[articlePicked_id][1] = np.sqrt(variance)

        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.random.normal(*self.UserArmPriors[article.id])
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class TSMultiArmedBandit:
    def __init__(self, num_arm):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TSStruct(self.num_arm)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean


