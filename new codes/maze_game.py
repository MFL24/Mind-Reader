import numpy as np
import pygame
import time
from gadgets import Player,Wall,Flag
import multiprocessing as mp


def maze_game(queue_action):
    pygame.init()
    vec = pygame.math.Vector2

    # Create a window
    window_size = (800, 600) # set window size
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Maze Game")

    PlayerInitialPosition = vec(50,500)
    P1 = Player((50,50),PlayerInitialPosition,window,color=(255,255,0))


    wall_size = ([800,20],[20,800],[20,800],[800,20],
                    [500,20],[500,20],[500,20])
    wall_position = ([0,580],[0,0],[780,0],[0,0],
                        [0,450],[0,100],[300,300])

    F = Flag((30,30),[50,50])

    Walls = pygame.sprite.Group()
    Gadgets = pygame.sprite.Group()    
    Interactive = pygame.sprite.Group()   
    for i in range(len(wall_position)):
        w_tempt = Wall(wall_size[i],wall_position[i],(0,0,0))
        Walls.add(w_tempt)
        Gadgets.add(w_tempt)
    Gadgets.add(P1)
    Gadgets.add(F)
    Interactive.add(F)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        window.fill((255,255,255))
        if not queue_action.empty():
            action = queue_action.get()
            last_movement = P1.move(action) # Right, Left, Up, Down, None
        else:
            last_movement = P1.move('Still')
            
        P1.update(last_movement,Walls)
        P1.win(Interactive,'./Maze_Game/winning.png')
        for entity in Gadgets:
                window.blit(entity.surf, entity.rect)
        pygame.display.update()

