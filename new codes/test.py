import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import find_channel_index
from scipy.signal import cheby2, cheb2ord, filtfilt,  bode, freqz
import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self,size,position,speed,img_sc=False):
        '''
    
        create main character according to an image or just a rectangle

        ------------
        Paramaters: 
        size: tuple of int
        describing the size of the player
        img_sc: str 
        image source of the player, default as False
        position: tuple of int
        initial position of the player 
        speed: int
        moving distance by one command
        
        ------------
        Return:
        player.surf: pygame.surface 
        the surface of the player
        player.rect: pygame.rect 
        rectangle of the player
          
        '''
        super().__init__() 

        if img_sc:
            self.surf = pygame.image.load(img_sc)
            self.surf = pygame.transform.scale(self.surf,size)
        else:
            self.surf = pygame.Surface(size)
            self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = position)
        self.set_SurfPosition(position)
        self._speed = speed
        
    @property
    def SurfPosition(self):
        return self._SurfPosition
    
    @SurfPosition.setter
    def set_SurfPosition(self,position):
        self._SurfPosition = position
        
    def _collision_detect():
        def Wrapper(func):
            def InnerWrapper(self,displacement,wall):
                current_position = self.SurfPosition
                test_rect = self.surf.get_rect(center = current_position+displacement)
                if test_rect.colliderect(wall):
                    print('Collide')
                    return 
                func(self,displacement,wall)
            return InnerWrapper  
        return Wrapper
    
    @_collision_detect()
    def move(self,displacement,wall):
        self.set_SurfPosition(self.SurfPosition + displacement)
        self.rect = self.rect.move(*displacement)
    
    
class Wall(pygame.sprite.Sprite):
    def __init__(self,size,position):
        super().__init__()
        self.surf = pygame.Surface(size)
        self.surf.fill((255,0,0))
        self.rect = self.surf.get_rect(center = position)
        self.set_SurfPosition(position)
        
    @property
    def SurfPosition(self):
        return self._SurfPosition
    
    @SurfPosition.setter
    def set_SurfPosition(self,position):
        self._SurfPosition = position


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
    print(figure_rect.x)
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
                old_figure_rect = [figure_rect.x,figure_rect.y]
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
                print('collide')
                position_fig = old_position_fig.copy()
                figure_rect.x = old_figure_rect[0]
                figure_rect.y = old_figure_rect[1]
                
            window.blit(figure,position_fig)

            pygame.draw.rect(window, (0,0,0), wall)
            # Update the display
            pygame.display.flip()
            # pygame.time.wait(1000)