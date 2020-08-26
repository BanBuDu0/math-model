import random
import os


class LifeGame:
    def __init__(self):
        self.width = 50
        self.height = 50
        self.screen_old = []
        self.screen = []

    def init_screen(self):
        self.screen = [['#' if random.random() > 0.8 else ' ' for _ in range(self.width)]
                       for _ in range(self.height)]
        self.screen_old = self.screen.copy()
        self.print_screen()

    def get_cell(self, _h, _w):
        if 0 <= _h < self.height and 0 <= _w < self.height:
            return self.screen_old[min(_h, self.height - 1)][min(_w, self.width - 1)]
        else:
            return ' '

    def get_nearby_cells_count(self, h, w):
        nearby = [self.get_cell(h + dy, w + dx) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        return len(list(filter(lambda x: x == '#', nearby)))

    def print_screen(self):
        os.system('cls')
        for i in self.screen:
            for j in i:
                print(j, end=' ')
            print('|')
        print()

    def get_new_cell(self, h, w):
        count = self.get_nearby_cells_count(h, w)
        # print(count)
        if self.screen_old[h][w] == '#':
            if count == 3 or count == 2:
                return '#'
            else:
                return ' '
        else:
            if count == 3:
                return '#'
            else:
                return self.screen_old[h][w]

    def update(self):
        self.screen = [[self.get_new_cell(h, w) for w in range(self.width)] for h in range(self.height)]

    def loop(self):
        self.update()
        self.print_screen()

    def run(self):
        self.init_screen()
        self.screen_old = self.screen.copy()
        self.loop()
        while self.screen_old != self.screen:
            self.screen_old = self.screen.copy()
            self.loop()
        print('End')


if __name__ == "__main__":
    # Start()
    l = LifeGame()
    l.run()
