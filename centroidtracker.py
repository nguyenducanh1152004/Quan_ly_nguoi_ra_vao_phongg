import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=80, maxDistance=50, history_length=10):
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.objectPaths = OrderedDict()  
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.history_length = history_length  

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.objectPaths[self.nextObjectID] = [centroid]  
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.objectPaths[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                prev_centroid = self.objects[objectID]
                new_centroid = inputCentroids[col]

                if dist.euclidean(prev_centroid, new_centroid) > 3:
                    self.objects[objectID] = new_centroid
                    self.disappeared[objectID] = 0
                    self.objectPaths[objectID].append(new_centroid)
                    if len(self.objectPaths[objectID]) > self.history_length:
                        self.objectPaths[objectID].pop(0)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])) - usedRows
            unusedCols = set(range(0, D.shape[1])) - usedCols

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    new_centroid = inputCentroids[col]
                    close_existing = any(
                        dist.euclidean(new_centroid, self.objectPaths[objID][-1]) < self.maxDistance * 1.5
                        for objID in self.objectPaths
                    )
                    if not close_existing:
                        self.register(new_centroid)

        return self.objects
