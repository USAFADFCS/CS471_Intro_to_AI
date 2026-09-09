import pygame
import sys
import heapq
import random

# --- INITIALIZATION & CONSTANTS ---
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("USAF AI Command Center - Educational Simulator")

# Colors (USAF Theme)
USAF_BLUE = (0, 48, 143)
AIR_FORCE_YELLOW = (255, 203, 5)
DARK_GRAY = (30, 30, 30)
LIGHT_GRAY = (200, 200, 200)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (220, 20, 60)
RADAR_GREEN = (0, 150, 0)

# Fonts
font_title = pygame.font.SysFont("Courier", 36, bold=True)
font_text = pygame.font.SysFont("Courier", 18)
font_small = pygame.font.SysFont("Courier", 14)

# --- HELPER FUNCTIONS ---
def draw_text(text, font, color, x, y, center=False):
    img = font.render(text, True, color)
    if center:
        rect = img.get_rect(center=(x, y))
        screen.blit(img, rect)
    else:
        screen.blit(img, (x, y))

def wrap_text(text, font, color, x, y, max_width):
    words = text.split(' ')
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        fw, fh = font.size(' '.join(current_line))
        if fw > max_width:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    
    for i, line in enumerate(lines):
        draw_text(line, font, color, x, y + (i * fh))

# --- SCENES ---
class MenuScene:
    def __init__(self):
        self.options = [
            ("Module 1: Search & Problem Formulation (A*)", SearchScene),
            ("Module 2: MDP & Reinforcement Learning", RLScene),
            ("Module 3: Probabilistic Tracking (HMMs)", HMMScene),
            ("Exit Command Center", None)
        ]
    
    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(len(self.options)):
                    rect = pygame.Rect(WIDTH//2 - 250, 250 + i*70, 500, 50)
                    if rect.collidepoint(mx, my):
                        if self.options[i][1] is None:
                            pygame.quit()
                            sys.exit()
                        return self.options[i][1]()
        return self

    def update(self):
        pass

    def draw(self, screen):
        screen.fill(DARK_GRAY)
        draw_text("USAF AI COMMAND CENTER", font_title, AIR_FORCE_YELLOW, WIDTH//2, 100, center=True)
        draw_text("Select a Training Module based on the Syllabus", font_text, WHITE, WIDTH//2, 150, center=True)
        
        for i, (text, _) in enumerate(self.options):
            rect = pygame.Rect(WIDTH//2 - 250, 250 + i*70, 500, 50)
            pygame.draw.rect(screen, USAF_BLUE, rect)
            pygame.draw.rect(screen, AIR_FORCE_YELLOW, rect, 2)
            draw_text(text, font_text, WHITE, WIDTH//2, 275 + i*70, center=True)

# ---------------------------------------------------------
# MODULE 1: SEARCH (Lessons 4-7)
# ---------------------------------------------------------
class SearchScene:
    def __init__(self):
        self.cols, self.rows = 20, 15
        self.cell_size = 40
        self.grid_offset_x = 50
        self.grid_offset_y = 50
        self.obstacles = set()
        self.start = (0, 0)
        self.goal = (19, 14)
        self.path = []
        self.frontier = []
        self.explored = set()
        
        self.desc = "LESSONS 4-7: Informed Search & Problem Formulation. Define states (grid cells), actions (moves), and path costs. A* Search uses heuristics (distance to goal) to find the optimal flight path for a UAV avoiding SAM sites."

    def get_neighbors(self, node):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in self.obstacles]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance

    def run_astar(self):
        # Immediate calculation for visualization
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        
        self.explored = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            self.explored.add(current)
            
            if current == self.goal:
                break
                
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        # Reconstruct path
        self.path = []
        curr = self.goal
        if curr in came_from:
            while curr != self.start:
                self.path.append(curr)
                curr = came_from[curr]
            self.path.append(self.start)
            self.path.reverse()

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return MenuScene()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                gx = (mx - self.grid_offset_x) // self.cell_size
                gy = (my - self.grid_offset_y) // self.cell_size
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    if (gx, gy) != self.start and (gx, gy) != self.goal:
                        if (gx, gy) in self.obstacles:
                            self.obstacles.remove((gx, gy))
                        else:
                            self.obstacles.add((gx, gy))
                        self.run_astar()
        return self

    def update(self):
        pass

    def draw(self, screen):
        screen.fill(DARK_GRAY)
        draw_text("OPERATION: SAFE CORRIDOR (A* Search)", font_title, AIR_FORCE_YELLOW, 20, 10)
        wrap_text(self.desc, font_small, WHITE, 20, 60, WIDTH - 40)
        draw_text("Click grid to add/remove Enemy SAM Sites (Obstacles). Press ESC to return.", font_small, AIR_FORCE_YELLOW, 20, HEIGHT - 30)

        # Draw Grid
        for x in range(self.cols):
            for y in range(self.rows):
                rect = pygame.Rect(self.grid_offset_x + x*self.cell_size, self.grid_offset_y + y*self.cell_size + 50, self.cell_size, self.cell_size)
                
                if (x, y) == self.start:
                    pygame.draw.rect(screen, USAF_BLUE, rect)
                    draw_text("HQ", font_small, WHITE, rect.centerx, rect.centery, center=True)
                elif (x, y) == self.goal:
                    pygame.draw.rect(screen, GREEN, rect)
                    draw_text("OBJ", font_small, DARK_GRAY, rect.centerx, rect.centery, center=True)
                elif (x, y) in self.obstacles:
                    pygame.draw.rect(screen, RED, rect)
                    draw_text("SAM", font_small, WHITE, rect.centerx, rect.centery, center=True)
                elif (x, y) in self.path:
                    pygame.draw.rect(screen, AIR_FORCE_YELLOW, rect)
                elif (x, y) in self.explored:
                    pygame.draw.rect(screen, (50, 50, 80), rect)
                else:
                    pygame.draw.rect(screen, LIGHT_GRAY, rect, 1)

# ---------------------------------------------------------
# MODULE 2: MDP & RL (Lessons 11-15)
# ---------------------------------------------------------
class RLScene:
    def __init__(self):
        self.size = 6
        self.cell_size = 80
        self.grid_offset_x = 250
        self.grid_offset_y = 120
        self.values = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        self.policy = [[None for _ in range(self.size)] for _ in range(self.size)]
        
        self.goal = (5, 5) # Target (+100)
        self.hazards = {(2, 2): -100, (2, 3): -100, (4, 1): -50} # Turbulence/Threats
        
        self.gamma = 0.9 # Discount factor
        self.desc = "LESSONS 11-15: MDPs & Reinforcement Learning. Values represent expected cumulative reward. Value Iteration calculates the optimal 'Policy' (arrows) for an autonomous drone to reach the target (GREEN) while avoiding hazards (RED). Press SPACE to step through Value Iteration."

    def step_value_iteration(self):
        new_values = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        actions = [(0, -1, 'U'), (0, 1, 'D'), (-1, 0, 'L'), (1, 0, 'R')]
        
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) == self.goal:
                    new_values[y][x] = 100.0
                    continue
                if (x, y) in self.hazards:
                    new_values[y][x] = self.hazards[(x, y)]
                    continue
                
                max_v = -float('inf')
                best_a = None
                for dx, dy, a_name in actions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        # Assuming deterministic transitions for visualization simplicity
                        v = 0 + self.gamma * self.values[ny][nx]
                    else:
                        v = -1 + self.gamma * self.values[y][x] # Wall bump penalty
                    
                    if v > max_v:
                        max_v = v
                        best_a = a_name
                
                new_values[y][x] = max_v
                self.policy[y][x] = best_a
        self.values = new_values

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return MenuScene()
                if event.key == pygame.K_SPACE:
                    self.step_value_iteration()
        return self

    def update(self): pass

    def draw_arrow(self, surface, color, x, y, direction):
        length = 20
        if direction == 'U': start, end = (x, y+length), (x, y-length)
        elif direction == 'D': start, end = (x, y-length), (x, y+length)
        elif direction == 'L': start, end = (x+length, y), (x-length, y)
        elif direction == 'R': start, end = (x-length, y), (x+length, y)
        else: return
        pygame.draw.line(surface, color, start, end, 3)
        pygame.draw.circle(surface, color, end, 5)

    def draw(self, screen):
        screen.fill(DARK_GRAY)
        draw_text("OPERATION: AUTONOMOUS FLIGHT (MDP / Value Iteration)", font_title, AIR_FORCE_YELLOW, 20, 10)
        wrap_text(self.desc, font_small, WHITE, 20, 60, WIDTH - 40)
        draw_text("Press SPACE to perform one iteration of the Bellman Update. ESC to return.", font_small, AIR_FORCE_YELLOW, 20, HEIGHT - 30)

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(self.grid_offset_x + x*self.cell_size, self.grid_offset_y + y*self.cell_size, self.cell_size, self.cell_size)
                
                if (x, y) == self.goal:
                    pygame.draw.rect(screen, GREEN, rect)
                    draw_text("+100", font_text, DARK_GRAY, rect.centerx, rect.centery, center=True)
                elif (x, y) in self.hazards:
                    pygame.draw.rect(screen, RED, rect)
                    draw_text(str(self.hazards[(x,y)]), font_text, WHITE, rect.centerx, rect.centery, center=True)
                else:
                    val_color = min(255, max(0, int((self.values[y][x] / 100.0) * 255)))
                    pygame.draw.rect(screen, (val_color, val_color, val_color), rect)
                    pygame.draw.rect(screen, LIGHT_GRAY, rect, 1)
                    
                    # Draw value
                    val_str = f"{self.values[y][x]:.1f}"
                    draw_text(val_str, font_small, AIR_FORCE_YELLOW, rect.left + 5, rect.top + 5)
                    
                    # Draw policy arrow
                    if self.policy[y][x]:
                        self.draw_arrow(screen, USAF_BLUE, rect.centerx, rect.centery, self.policy[y][x])

# ---------------------------------------------------------
# MODULE 3: PROBABILITY & HMM (Lessons 18-23)
# ---------------------------------------------------------
class HMMScene:
    def __init__(self):
        self.num_sectors = 10
        # The hidden actual location of the stealth plane
        self.true_pos = 0 
        
        # The AI's belief distribution (starts uniform)
        self.belief = [1.0 / self.num_sectors for _ in range(self.num_sectors)]
        
        self.desc = "LESSONS 18-23: Probability & HMMs. The Stealth Aircraft (Hidden State) moves. Radar provides noisy observations. The AI updates its belief distribution (green bars) using Bayes' Rule and Markov transitions to estimate the plane's true location."

    def update_hmm(self):
        # 1. Hidden State Transition: Plane moves Right (80%), Stays (20%)
        if random.random() < 0.8 and self.true_pos < self.num_sectors - 1:
            self.true_pos += 1
            
        # 2. Time Update (AI applies Transition Model)
        new_belief = [0.0] * self.num_sectors
        for i in range(self.num_sectors):
            # Probability it stayed
            new_belief[i] += self.belief[i] * 0.2
            # Probability it moved from the left
            if i > 0:
                new_belief[i] += self.belief[i-1] * 0.8
        self.belief = new_belief

        # 3. Generate Noisy Observation based on Sensor Model
        # Sensor reads true position 60% of time, +/- 1 position 20% each
        obs = self.true_pos
        r = random.random()
        if r < 0.2: obs = max(0, self.true_pos - 1)
        elif r > 0.8: obs = min(self.num_sectors - 1, self.true_pos + 1)

        # 4. Measurement Update (AI applies Bayes' Rule using observation)
        for i in range(self.num_sectors):
            # P(obs | state=i)
            if i == obs: p_z = 0.6
            elif abs(i - obs) == 1: p_z = 0.2
            else: p_z = 0.01 # small chance of random noise anywhere
            
            self.belief[i] = self.belief[i] * p_z

        # Normalize belief
        total = sum(self.belief)
        self.belief = [b / total for b in self.belief]

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return MenuScene()
                if event.key == pygame.K_SPACE:
                    self.update_hmm()
        return self

    def update(self): pass

    def draw(self, screen):
        screen.fill(DARK_GRAY)
        draw_text("OPERATION: STEALTH TRACKER (Hidden Markov Models)", font_title, AIR_FORCE_YELLOW, 20, 10)
        wrap_text(self.desc, font_small, WHITE, 20, 60, WIDTH - 40)
        draw_text("Press SPACE to advance 1 Time Step (Plane moves, Radar pings). ESC to return.", font_small, AIR_FORCE_YELLOW, 20, HEIGHT - 30)

        # Draw Grid
        bar_width = 70
        start_x = (WIDTH - (self.num_sectors * (bar_width + 10))) // 2
        
        for i in range(self.num_sectors):
            x = start_x + i * (bar_width + 10)
            
            # Draw AI Belief (Probability distribution)
            prob = self.belief[i]
            bar_height = int(prob * 300)
            rect = pygame.Rect(x, 500 - bar_height, bar_width, bar_height)
            pygame.draw.rect(screen, RADAR_GREEN, rect)
            
            # Draw Sector Labels
            draw_text(f"Sec {i}", font_small, WHITE, x + bar_width//2, 520, center=True)
            draw_text(f"{prob*100:.1f}%", font_small, AIR_FORCE_YELLOW, x + bar_width//2, 480 - bar_height, center=True)

            # Draw True Plane Location (Hidden from AI, shown for educational purpose)
            if i == self.true_pos:
                plane_rect = pygame.Rect(x + 10, 560, 50, 20)
                pygame.draw.rect(screen, USAF_BLUE, plane_rect)
                draw_text("PLANE", font_small, WHITE, x + bar_width//2, 570, center=True)
                draw_text("(Hidden State)", font_small, LIGHT_GRAY, x + bar_width//2, 595, center=True)

# --- MAIN LOOP ---
def main():
    clock = pygame.time.Clock()
    current_scene = MenuScene()

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        next_scene = current_scene.handle_events(events)
        if next_scene is not current_scene:
            current_scene = next_scene

        current_scene.update()
        current_scene.draw(screen)
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()