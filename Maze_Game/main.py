import numpy as np
import pygame
import time


if __name__ == '__main__':
    # initalize
    pygame.init()

    # Create a window
    window_size = (800, 600) # set window size
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Maze Game")

    # Load an image into a surface
    figure = pygame.image.load('./Maze_Game/test.png') # load test figure image
    figure = pygame.transform.scale(figure,(30,50)) # scale the figure to smaller size
    figure_rect = figure.get_rect() # obtain the rect object of the figure
    
    # Set initial position of figure
    position_fig = [100, 100]
    
    # Set initial position of walls
    position_wall = [200,200]
    
    wall = pygame.Rect([200,200,50,50])
    
    # redefine the position of rectangle for collision detection
    figure_rect.x = position_fig[0]
    figure_rect.y = position_fig[1]
    
    # Define movement speed
    speed = 10

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == 771: # 771 is the alphabet
                old_position_fig = position_fig.copy()
                if event.text == 'a':
                    position_fig[0] -= speed 
                    figure_rect.x -= speed                   
                elif event.text == 'd':
                    position_fig[0] += speed  
                    figure_rect.x += speed   
                elif event.text == 'w':
                    position_fig[1] -= speed 
                    figure_rect.y -= speed   
                elif event.text == 's':
                    position_fig[1] += speed 
                    figure_rect.y += speed   

            # Clear the window
            window.fill((255, 255, 255))  # Fill with white color

            # Blit the surface at the updated position
            if figure_rect.colliderect(wall):
                print(id(old_position_fig),id(position_fig))
                print('old: {}; new: {}'.format(old_position_fig,position_fig))
                position_fig = old_position_fig
                
            window.blit(figure,position_fig)

            pygame.draw.rect(window, (0,0,0), wall)
            # Update the display
            pygame.display.flip()
            # pygame.time.wait(1000)