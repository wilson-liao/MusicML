from pygame import mixer
import pygame

class MusicPlayer:
    def __init__(self, path):
        pygame.init()
        mixer.init()
        self.path = path
        self.clock = pygame.time.Clock()

    def play_music(self):
        mixer.music.load(self.path)
        mixer.music.play()
        while pygame.mixer.music.get_busy():
            self.clock.tick(30)
        pygame.quit()







