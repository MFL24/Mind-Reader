import numpy as np
import pygame



if __name__ == '__main__':
    pygame.init() 
    
    # CREATING CANVAS 
    canvas = pygame.display.set_mode((500, 500)) 

    color = (255,255,255) 
    rect_color = (255,0,0) 

    rect_holder = pygame.Rect(50,50,60,60)
    # TITLE OF CANVAS 
    pygame.display.set_caption("Show Image") 
    
    img = pygame.image.load('./Maze_Game/test.png')
    print(type(img))
    
    exit = False
    
    while not exit: 
        canvas.fill(color) 
        
        for event in pygame.event.get(): 
            # print(event)
            if event.type == pygame.QUIT: 
                exit = True
            elif event.type == 771:
                if event.text == 'a':
                    print('a pressed')
                    rect_holder = rect_holder.move(-10,0)
                    canvas.scroll(-10,0)
                elif event.text == 'd':
                    print('d pressed')
                    rect_holder = rect_holder.move(10,0)
                elif event.text == 'w':
                    print('w pressed')
                    rect_holder = rect_holder.move(0,-10)
                elif event.text == 's':
                    print('s pressed')
                    rect_holder = rect_holder.move(0,10) 
                    
        canvas.blit(img,(0,0)) 
        pygame.draw.rect(canvas, rect_color, rect_holder) 
        pygame.display.update() 