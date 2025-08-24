import json
import statistics as st
import os
import numpy as np
import pygame
from matplotlib import pyplot as plt
from pygame.locals import *
from Matrix import Matrix
from genetic_algorithm import create_initial_genes, generate_next_generation
from RedeNeuronal import Network

os.makedirs('../results', exist_ok=True)

auto_running = True

NUM_OF_GENES = 200
INPUT_SIZE = 4
HIDDEN_LAYER_SIZE = 6
OUTPUT_SIZE = 4
NUM_OF_GENERATIONS = 20  # -> Number of generations
TOP_GENES = 80  # -> Top genes to be selected for crossover
MUTATION_RATE = 0.05
MAX_LIFE_GENERATION = 100
CROSSOVER_K_POINTS = 1
MAX_PASSOS = 30


genes = create_initial_genes()
avg_fitness_score_list = []
avg_garbage_left_list = []

pygame.init()
iconSize = (100, 100)  # (y, x)
totalWidthAndHeight = 5
windowWidth = iconSize[0] * totalWidthAndHeight
windowHeight = iconSize[0] * totalWidthAndHeight

pygame.init()

screen = pygame.display.set_mode((windowWidth + 200, windowHeight), pygame.HWSURFACE)
screen.fill((255, 255, 255))
boardGame = pygame.Surface((windowWidth, windowHeight))
screen_text = pygame.Surface((200, windowWidth))
screen_text.fill((255, 255, 255))

pygame.display.set_caption('Iron Garbage')
_image_iron_man = pygame.image.load("../textures/iron-man.png").convert_alpha()
_image_iron_man = pygame.transform.scale(_image_iron_man, (iconSize[0], iconSize[1]))
_image_garbage = pygame.image.load("../textures/garbage.jpg").convert_alpha()
_image_garbage = pygame.transform.scale(_image_garbage, (iconSize[0], iconSize[1]))

pygame.font.init()
font = pygame.font.Font(None, 36)

for generation in range(NUM_OF_GENERATIONS):
    print('Generation: ', generation, 'started')
    end_game = False
    fitness_scores = []
    garbage_left = []
    for gene in genes:
        m = Matrix(windowHeight, windowWidth, iconSize)
        m.fillIronMatrix(0, 0)

        garbage_positions = [(0, 3), (1, 1), (1, 4), (2, 2), (3, 1), (3, 3), (4, 4)]
        m.spwanGarbage(garbage_positions)

        passo = 0

        while not end_game and passo < MAX_PASSOS:
            screen.fill((255, 255, 255))  # Clear the screen each frame
            screen_text.fill((255, 255, 255))

            inputs = m.get_neighboors()  # entradas relevantes do ambiente
            inputs = np.array(inputs)  # converter lista para um array
            #print("Shape of inputs:", inputs.shape)  # verificar se os inputs estao na shape que devem
            movimento = gene.move(inputs)

            prev_value = None

            if movimento == 3:
                prev_value = m.fillIronMatrix(m.yIron, m.xIron - 1)
            elif movimento == 1:
                prev_value = m.fillIronMatrix(m.yIron, m.xIron + 1)
            elif movimento == 2:
                prev_value = m.fillIronMatrix(m.yIron - 1, m.xIron)
            elif movimento == 0:
                prev_value = m.fillIronMatrix(m.yIron + 1, m.xIron)

            if prev_value == 'garbage':
                gene.catch_garbage()
                m.decrement_left_garbage()

            m.drawMatrix(boardGame, _image_iron_man, _image_garbage, iconSize)
            screen.blit(boardGame, (200, 0))

            screen_text = font.render(f'Geração: {generation}', True, (0, 0, 0))
            screen.blit(screen_text, (10,10))

            fitness_text = font.render(f'Pontuação: {gene.get_fitness_score()}', True, (0, 0, 0))
            screen.blit(fitness_text, (10, 50))

            lixo_em_falta = font.render(f'Lixo em falta: {m.get_left_garbage()}', True, (0, 0, 0), (255, 255, 255))
            screen.blit(lixo_em_falta, (10, 90))

            pygame.display.flip()

            pygame.display.update()

            #inputs = m.get_neighboors()
            if m.get_left_garbage() == 0:
                end_game = True
                break

            passo += 1
            #print("Passo: ", passo)

        fitness_score = gene.get_fitness_score()
        fitness_scores.append(fitness_score)
        garbage_left.append(m.get_left_garbage())
        gene.reset_fitness_score()

    top_genes = sorted(genes, key=lambda gene: gene.get_fitness_score(), reverse=True)[:TOP_GENES]

    genes = generate_next_generation(top_genes)

    avg_fitness_score = sum(fitness_scores) / len(fitness_scores)
    avg_garbage_left = sum(garbage_left) / len(garbage_left)

    avg_fitness_score_list.append(avg_fitness_score)
    avg_garbage_left_list.append(avg_garbage_left)
    print(f'Generation {generation} avg fitness score: {avg_fitness_score}')
    print(f'Generation {generation} avg garbage left: {avg_garbage_left}')

    #genes = generate_next_generation(genes)

save = {"last_generation_dna": [gene.get_dna() for gene in genes],
    "avg_fitness_score": avg_fitness_score_list,
    "avg_garbage_left": avg_garbage_left_list}

print("Dados a serem salvos:")
print(save)


with open(f'../results/generation_1.json', 'w') as f:
    json.dump(save, f)
    print('Resultados guardados')


generations = list(range(NUM_OF_GENERATIONS))
avg_fitness_score = avg_fitness_score_list

plt.figure(figsize=(10, 6))
plt.plot(generations, avg_fitness_score, marker='o', linestyle='-', color='b')
plt.title('Comparação entre Geração e Pontuação Média de Aptidão')
plt.xlabel('Geração')
plt.ylabel('Pontuação Média de Aptidão')
plt.savefig('aptidão.png')
plt.show()