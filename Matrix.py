import numpy as np
import pygame


class Matrix(object):

    def __init__(self, height, width, iconSize):
        self._height = height
        self._width = width
        self._iconSize = iconSize
        self._matrixRows = self._height // iconSize[0]
        self._matrixCols = self._width // iconSize[1]
        self.matrix = np.ndarray((self._matrixRows, self._matrixCols), dtype=object)
        self.matrix[:, :] = 'empty'
        self.yIron = None
        self.xIron = None
        self._left_garbage = 7

    def get_left_garbage(self):
        return self._left_garbage

    def decrement_left_garbage(self):
        self._left_garbage -= 1

    def fillEmptyMatrix(self, y, x):
        self.matrix[y, x] = 'empty'

    def fillIronMatrix(self, y, x):
        if self.yIron is not None:
            self.fillEmptyMatrix(self.yIron, self.xIron)

        # update iron position
        # * if it goes to wall then go to the other side
        if x < 0:
            x = self._matrixCols - 1
        if x >= self._matrixCols:
            x = 0
        if y < 0:
            y = self._matrixRows - 1
        if y >= self._matrixRows:
            y = 0

        # save the previous value -> empty or garbage
        prev_value = self.matrix[y, x]

        self.matrix[y, x] = 'iron'
        self.yIron = y
        self.xIron = x

        return prev_value

    def fillGarbageMatrix(self, y, x):
        self.matrix[y, x] = 'garbage'

    # def spwanGarbageRandom(self, numberOfTimes):
    #     for _ in range(numberOfTimes):
    def spwanGarbage(self, garbage_positions):
        for (i, j) in garbage_positions:
            self.matrix[i, j] = 'garbage'
            # while True:
            #     x = random.randrange(0, self._matrixCols, 1)
            #     y = random.randrange(0, self._matrixRows, 1)
            #
            #     if self.matrix[y, x] != 'empty':
            #         continue
            #     else:
            #         # print(f'Appel->{[self.y, self.x]}')
            #         # print(f'Snake->{tmp}')
            #         self.matrix[y, x] = 'garbage'
            #         break

    def drawMatrix(self, surface, _image_iron_man, _image_garbage, iconSize):
        surface.fill((255, 255, 255))

        for y in range(self._matrixRows):
            for x in range(self._matrixCols):
                if self.matrix[y, x] == 'iron':
                    surface.blit(_image_iron_man, (x * iconSize[1], y * iconSize[0]), (0, 0, iconSize[1], iconSize[0]))
                elif self.matrix[y, x] == 'garbage':
                    surface.blit(_image_garbage, (x * iconSize[1], y * iconSize[0]), (0, 0, iconSize[1], iconSize[0]))

        for y in range(0, self._height, self._iconSize[0]):
            pygame.draw.line(surface, (0, 0, 0), (0, y), (self._width - 1, y), 2)

        for x in range(0, self._width, self._iconSize[1]):
            pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, self._height - 1), 2)

    pass

    def get_neighboors(self):
        vizinhos = []
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in direcoes:
            ny, nx = self.yIron + dy, self.xIron + dx
            if 0 <= ny < self.matrix.shape[0] and 0 <= nx < self.matrix.shape[1]:
                if self.matrix[ny, nx] == 'empty':
                    vizinhos.append(0)
                elif self.matrix[ny, nx] == 'iron':
                    vizinhos.append(1)
                elif self.matrix[ny, nx] == 'garbage':
                    vizinhos.append(2)
            else:
                vizinhos.append(-1)
        return vizinhos