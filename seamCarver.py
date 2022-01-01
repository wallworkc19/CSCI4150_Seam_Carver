# Caleb Wallwork
# HW2: Seam Carving for Content-Aware Image Resizing
# CSCI 4150

import numpy as np
import cv2


class seamCarver:
    # initialize
    def __init__(self, file, height, width, protectMask = '', objectMask = ''):
        self.file = file
        self.heightOut = height
        self.widthOut = width

        # read the image
        self.imageIn = cv2.imread(file).astype(np.float64)
        self.heightIn, self.widthIn = self.imageIn.shape[: 2]
        self.imageOut = np.copy(self.imageIn)

        # object removal
        self.object = (objectMask != '')
        if self.object:
            self.mask = cv2.imread(objectMask, 0).astype(np.float64)
            self.protect = False

        # image resize
        else:
            self.protect = (protectMask != '')
            if self.protect:
                self.mask = cv2.imread(protectMask, 0).astype(np.float64)

        # forward energy map
        self.kernelA = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernelBL = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernelBR = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # constant area
        self.c = 1000

        # start
        self.start()

    
    def start(self):
        if self.object:
            self.objectRemove()
        else:
            self.seamCarve()

    def objectRemove(self):
        rotate = False
        objectH, objectW = self.getDimension()
        if objectH < objectW:
            self.imageOut = self.rotateImage(self.imageOut, 1)
            self.mask = self.rotateMask(self.mask, 1)
            rotate = True

        while len(np.where(self.mask[:, :] > 0)[0]) > 0:
            map = self.calcMap()
            map[np.where(self.mask[:, :] > 0)] *= -self.c
            cMap = self.cumulativeMapForward(map)
            seamX = self.getSeam(cMap)
            self.deleteSeam(seamX)
            self.deleteMaskSeam(seamX)

        if not rotate:
            pixels = self.widthIn - self.imageOut.shape[1]
        else:
            pixels = self.heightIn - self.imageOut.shape[1]

        self.seamsInsert(pixels)
        if rotate:
            self.imageOut = self.rotateImage(self.imageOut, 0)


    def seamCarve(self):
        dRow, dCol = int(self.heightOut - self.heightIn), int(self.widthOut - self.widthIn)

        if dcol < 0:
            self.seamsRemove(dCol * -1)

        elif dCol > 0:
            self.seamsInsert(dCol)

        if dRow < 0:
            self.imageOut = self.rotateImage(self.imageOut, 1)
            if self.protect:
                self.mask = self.rotateMask(self.mask, 1)
            self.seamsRemove(dRow * -1)
            self.imageOut = self.rotateImage(self.imageOut, 0)

        elif dRow > 0:
            self.imageOut = self.rotateImage(self.imageOut, 1)
            if self.protect:
                self.mask = self.rotateMask(self.mask, 1)
            self.seamsInsert(dRow)
            self.imageOut = self.rotateImage(self.imageOut, 0)

    def getSeam(self, cMap):
        x, y = cMap.shape
        out = np.zeros((x,), dtype = np.uint32)
        out[-1] = np.argmin(cMap[-1])
        for r in range(x - 2, -1, -1):
            z = out[r + 1]
            if z == 0:
                out[r] = np.argmin(cMap[r, : 2])
            else:
                out[r] = np.argmin(cMap[r, z - 1: min(z + 2, y - 1)]) + z - 1
        return out


    def deleteSeam(self, seamX):
        x, y = self.imageOut.shape[: 2]
        out = np.zeros((x, y - 1, 3))
        for r in range(x):
            c = seamX[r]
            out[r, :, 0] = np.delete(self.imageOut[r, :, 0], [c])
            out[r, :, 1] = np.delete(self.imageOut[r, :, 1], [c])
            out[r, :, 2] = np.delete(self.imageOut[r, :, 2], [c])
        self.imageOut = np.copy(out)


    def addSeam(self, seamX):
        x, y = self.imageOut.shape[: 2]
        out = np.zeros((x, y + 1, 3))
        for r in range(x):
            c = seamX[r]
            for ch in range(3):
                if c == 0:
                    p = np.average(self.imageOut[r, c: c + 2, ch])
                    out[r, c, ch] = self.imageOut[r, c, ch]
                    out[r, c + 1, ch] = p
                    out[r, c + 1:, ch] = self.imageOut[r, c:, ch]
                else:
                    p = np.average(self.imageOut[r, c - 1: c + 1, ch])
                    out[r, : c, ch] = self.imageOut[r, : c, ch]
                    out[r, c, ch] = p
                    out[r, c + 1:, ch] = self.imageOut[r, c:, ch]
        self.imageOut = np.copy(out)


    def updateSeam(self, remaining_seams, current_seam):
        out = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            out.append(seam)
        return out

    def seamsRemove(self, pixels):
        if self.protect:
            for x in range(pixels):
                map = self.calcMap()
                map[np.where(self.mask > 0)] *= self.c
                cMap = self.cumulativeMapForward(map)
                seamX = self.getSeam(cMap)
                self.deleteSeam(seamX)
                self.deleteMaskSeam(seamX)
        else:
            for x in range(pixels):
                map = self.calcMap()
                cMap = self.cumulativeMapForward(map)
                seamX = self.getSeam(cMap)
                self.deleteSeam(seamX)


    def seamsInsert(self, pixels):
        if self.protect:
            tImage = np.copy(self.imageOut)
            tMask = np.copy(self.mask)
            seams = []

            for x in range(pixels):
                map = self.calcMap()
                map[np.where(self.mask[:, :] > 0)] *= self.c
                cMap = self.cumulativeMapBackward(map)
                seamX = self.getSeam(cMap)
                seams.append(seamX)
                self.deleteSeam(seamX)
                self.deleteMaskSeam(seamX)

            self.imageOut = np.copy(tImage)
            self.mask = np.copy(tMask)
            y = len(seams)
            for x in range(y):
                seam = seams.pop(0)
                self.addSeam(seam)
                self.addMaskSeam(seam)
                seams = self.updateSeam(seams, seam)
        else:
            tImage = np.copy(self.imageOut)
            seams = []

            for x in range(pixels):
                map = self.calcMap()
                cMap = self.cumulativeMapBackward(map)
                seamX = self.getSeam(cMap)
                seams.append(seamX)
                self.deleteSeam(seamX)

            self.imageOut = np.copy(tImage)
            y = len(seams)
            for x in range(y):
                seam = seams.pop(0)
                self.addSeam(seam)
                seams = self.updateSeam(seams, seam)


    def calcMap(self):
        b, g, r = cv2.split(self.imageOut)
        bEnergy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        gEnergy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        rEnergy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return bEnergy + gEnergy + rEnergy


    def cumulativeMapBackward(self, map):
        x, y = map.shape
        out = np.copy(map)
        for r in range(1, x):
            for c in range(y):
                out[r, c] = map[r, c] + np.amin(out[r - 1, max(c - 1, 0): min(c + 2, y - 1)])
        return out


    def cumulativeMapForward(self, map):
        matA = self.calcNeighbor(self.kernelA)
        matrixBL = self.calcNeighbor(self.kernelBL)
        matrixBR = self.calcNeighbor(self.kernelBR)

        x, y = map.shape
        out = np.copy(map)
        for r in range(1, x):
            for c in range(y):
                if c == 0:
                    right = out[r - 1, c + 1] + matA[r - 1, c + 1] + matrixBR[r - 1, c + 1]
                    up = out[r - 1, c] + matX[r - 1, c]
                    out[r, c] = map[r, c] + min(right, up)
                elif c == y - 1:
                    left = out[r - 1, c - 1] + matA[r - 1, c - 1] + matrixBL[r - 1, c - 1]
                    up = out[r - 1, c] + matX[r - 1, c]
                    out[r, c] = map[r, c] + min(left, up)
                else:
                    left = out[r - 1, c - 1] + matA[r - 1, c - 1] + matrixBL[r - 1, c - 1]
                    right = out[r - 1, c + 1] + matA[r - 1, c + 1] + matrixBR[r - 1, c + 1]
                    up = out[r - 1, c] + matA[r - 1, c]
                    out[r, c] = map[r, c] + min(left, right, up)
        return out


    def calcNeighbor(self, kernel):
        b, g, r = cv2.split(self.imageOut)
        out = np.absolute(cv2.filter2D(b, -1, kernel = kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel = kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel = kernel))
        return out


    def rotateImage(self, image, ccw):
        x, y, ch = image.shape
        out = np.zeros((y, x, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for r in range(x):
                    out[:, r, c] = image_flip[r, :, c]
        else:
            for c in range(ch):
                for r in range(x):
                    out[:, x - 1 - r, c] = image[r, :, c]
        return out


    def rotateMask(self, mask, ccw):
        x, y = mask.shape
        out = np.zeros((y, x))
        if ccw > 0:
            image_flip = np.fliplr(mask)
            for r in range(x):
                out[:, r] = image_flip[r, : ]
        else:
            for r in range(x):
                out[:, x - 1 - r] = mask[r, : ]
        return out


    def deleteMaskSeam(self, seamX):
        x, y = self.mask.shape
        out = np.zeros((x, y - 1))
        for r in range(x):
            c = seamX[r]
            out[r, : ] = np.delete(self.mask[r, : ], [c])
        self.mask = np.copy(out)


    def addMaskSeam(self, seamX):
        x, y = self.mask.shape
        out = np.zeros((x, y + 1))
        for r in range(x):
            c = seamX[r]
            if c == 0:
                p = np.average(self.mask[r, c: c + 2])
                out[r, c] = self.mask[r, c]
                out[r, c + 1] = p
                out[r, c + 1: ] = self.mask[r, c: ]
            else:
                p = np.average(self.mask[r, c - 1: c + 1])
                out[r, : c] = self.mask[r, : c]
                out[r, c] = p
                out[r, c + 1: ] = self.mask[r, c: ]
        self.mask = np.copy(out)


    def getDimension(self):
        r, c = np.where(self.mask > 0)
        height = np.amax(r) - np.amin(r) + 1
        width = np.amax(c) - np.amin(c) + 1
        return height, width


    def save(self, file):
        cv2.imwrite(file, self.imageOut.astype(np.uint8))