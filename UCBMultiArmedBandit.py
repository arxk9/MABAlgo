import numpy as np

class UCBStruct:
    def __init__(self, num_arm):
        self.d = num_arm

        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.confidence_scale = 0.25
        # self.UserArmConfidence = np.zeros(self.d)

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1

        self.time += 1

    # def updateConfidences(self, pool_articles):
    #     for article in pool_articles:
    #         self.UserArmConfidence[article.id] = (self.UserArmMean[article.id] + sqrt(2 * log(self.time) / self.UserArmTrials[article.id]) 
    #                                                 if self.UserArmTrials[article.id] != 0 else float('inf'))

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        # self.updateConfidences(pool_articles)
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            # article_pta = self.UserArmMean[article.id] + self.UserArmConfidence[article.id]
            if self.UserArmTrials[article.id] == 0:
                return article
            article_pta = self.UserArmMean[article.id] + np.sqrt(2 * np.log(self.time) / self.UserArmTrials[article.id]) * self.confidence_scale
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class UCBMultiArmedBandit:
    def __init__(self, num_arm):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean


