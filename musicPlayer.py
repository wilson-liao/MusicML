from pygame import mixer
import pygame


class MusicPlayer:
    def __init__(self, path):
        pygame.init()
        # Print audio driver info
        print("Using audio driver:", pygame.mixer.get_init())
        
        # Initialize mixer with good quality settings
        mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.path = path
        self.clock = pygame.time.Clock()

    def play_music(self):
        mixer.music.load(self.path)
        mixer.music.play()
        while pygame.mixer.music.get_busy():
            self.clock.tick(60)
        pygame.quit()



if __name__ == "__main__":
    path = 'C:/Users/User/Desktop/wilson/MusicML/adl-piano-midi/Rock/Album Rock/Ace Frehley/Fractured Quantum.mid'
    mp = MusicPlayer(path)
    mp.play_music()