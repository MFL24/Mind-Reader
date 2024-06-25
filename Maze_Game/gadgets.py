import pygame
import numpy as np
import time

vec = pygame.math.Vector2


class Player(pygame.sprite.Sprite):
    SPEED = 0.5
    def __init__(self,size,position,screen,img_sc=False,color=(0,0,0)):
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
            self.surf.fill(color)
        self.size = size
        self.rect = self.surf.get_rect()
        self.__SurfPosition = position
        self._vel = vec(0,0)
        self.screen = screen
        
    @property
    def SurfPosition(self):
        '''
        get the position of the current surface
        '''
        return self.__SurfPosition
    
    @SurfPosition.setter
    def SurfPosition(self,position):
        '''
        setter method enabling changing the values of __SurfPosition
        '''
        if isinstance(position,pygame.math.Vector2):
            self.__SurfPosition = position
        else:
            raise Exception ('Unvalid data type')
        
    def move(self):
        '''
        control the movement of the player
        
        '''
        movement = None
        self._vel = vec(0,0)
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_a]:
            self._vel.x = -Player.SPEED
            movement = 'Left'
        if pressed_keys[pygame.K_d]:
            self._vel.x = Player.SPEED
            movement = 'Right'
        if pressed_keys[pygame.K_w]:
            self._vel.y = -Player.SPEED
            movement = 'Up'
        if pressed_keys[pygame.K_s]:
            self._vel.y = Player.SPEED 
            movement = 'Down'       
        
        self.SurfPosition += self._vel
        self.rect.topleft = self.SurfPosition
        return movement
    
    def update(self,movement,wall_group):
        '''
        check the collision of the player with walls 
        
        '''
        hits = pygame.sprite.spritecollide(self,wall_group,False)    
        if hits:
            print('Collision detected with {}'.format(movement))
            self._vel = vec(0,0)
            if movement == 'Left':
                self.SurfPosition.x = hits[0].rect.right+1
            elif movement == 'Right':
                self.SurfPosition.x = hits[0].rect.left-self.size[1]-1
            elif movement == 'Up':
                self.SurfPosition.y = hits[0].rect.bottom+1
            elif movement == 'Down':
                self.SurfPosition.y = hits[0].rect.top-self.size[0]-1
    
    def win(self,flag_group,win_imgsc):
        hits = pygame.sprite.spritecollide(self,flag_group,False)
        if hits:    
            bg = pygame.image.load(win_imgsc)
            bg = pygame.transform.scale(bg,(200,100))
            self.screen.blit(bg,(300,250))
            pygame.display.update()
            time.sleep(3)
            pygame.quit()
           
    
class Wall(pygame.sprite.Sprite):

    def __init__(self,size,position,color):
        '''
        create wall object
    
        Parameters:
        ------------
        size: tuple of int
        size of the rectangular walls
        position: tuple of int
        positions of the walls
        colors: tuple of int 
        RGB colors of the walls 
        
        Returns:
        -----------
        wall.surf: pygame.surface 
        the surface of the wall
        wall.rect: pygame.rect 
        rectangle of the wall
    
        '''
        super().__init__()
        self.surf = pygame.Surface(size)
        self.surf.fill(color)
        self.rect = self.surf.get_rect(topleft=position)
        self.__SurfPosition = position
        
    @property
    def SurfPosition(self):
        return self.__SurfPosition
    
    @SurfPosition.setter
    def SurfPosition(self,position):
        self.__SurfPosition = position
        
        
        
class Flag(pygame.sprite.Sprite):
    def __init__(self,size,position,color=(255,0,0)):
        '''
        flag object indicating ending of the game
        
        '''
        super().__init__()
        self.surf = pygame.Surface(size)
        self.surf.fill(color)
        self.rect = self.surf.get_rect(topleft=position)
        self.__SurfPosition = position
        
    @property
    def SurfPosition(self):
        return self.__SurfPosition
    
    @SurfPosition.setter
    def SurfPosition(self,position):
        self.__SurfPosition = position
        
        