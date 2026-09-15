import pygame
import sys
import heapq
import random
import io

# --- INITIALIZATION & CONSTANTS ---
pygame.init()

# Check for Matplotlib LaTeX engine
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Base Virtual Canvas Dimensions
V_WIDTH, V_HEIGHT = 1380, 920
virtual_surface = pygame.Surface((V_WIDTH, V_HEIGHT))

# Resizable Window (Scales cleanly to any monitor resolution)
screen = pygame.display.set_mode((V_WIDTH, V_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("USAF AI Command Center - CompSci 471 Simulator")

# Scaler State Variables
scale_factor = 1.0
offset_x = 0
offset_y = 0

# Colors (USAF Theme)
USAF_BLUE = (0, 48, 143)
AIR_FORCE_YELLOW = (255, 203, 5)
DARK_GRAY = (20, 20, 26)
PANEL_GRAY = (36, 36, 44)
LIGHT_GRAY = (200, 200, 200)
WHITE = (255, 255, 255)
GREEN = (40, 180, 90)
RED = (220, 50, 50)
WALL_COLOR = (85, 90, 100)
RADAR_GREEN = (0, 180, 60)
FRONTIER_COLOR = (100, 100, 150)
EXPLORED_COLOR = (50, 50, 80)
CYAN_ACCENT = (0, 220, 220)

# Fonts
font_title = pygame.font.SysFont("Courier", 32, bold=True)
font_subtitle = pygame.font.SysFont("Courier", 21, bold=True)
font_desc = pygame.font.SysFont("Courier", 18)
font_info = pygame.font.SysFont("Courier", 18, bold=True)
font_small = pygame.font.SysFont("Courier", 15)
font_tiny = pygame.font.SysFont("Courier", 12)

# --- LATEX & VIEWPORT ENGINE ---
math_surface_cache = {}

def get_latex_surface(latex_str, font_size=18, color=(255, 255, 255), dpi=130):
    """Renders LaTeX math to an anti-aliased, alpha-transparent Pygame surface."""
    cache_key = (latex_str, font_size, color, dpi)
    if cache_key in math_surface_cache:
        return math_surface_cache[cache_key]
    
    if HAS_MATPLOTLIB:
        try:
            fig = plt.figure(figsize=(0.05, 0.05), dpi=dpi)
            fig.patch.set_alpha(0.0)
            norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
            text_obj = fig.text(0, 0, f"${latex_str}$", fontsize=font_size, color=norm_color)
            
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text_obj.get_window_extent(renderer)
            
            w_in = max(0.1, (bbox.width + 12) / dpi)
            h_in = max(0.1, (bbox.height + 12) / dpi)
            fig.set_size_inches(w_in, h_in)
            text_obj.set_position((6.0 / (bbox.width + 12), 6.0 / (bbox.height + 12)))
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
            plt.close(fig)
            buf.seek(0)
            surf = pygame.image.load(buf).convert_alpha()
            math_surface_cache[cache_key] = surf
            return surf
        except Exception:
            pass

    # Unicode fallback if Matplotlib is not installed
    f = pygame.font.SysFont("Courier", font_size, bold=True)
    clean = latex_str.replace(r"\pi", "π").replace(r"\gamma", "γ").replace(r"\sum", "Σ")
    clean = clean.replace(r"\mathbb{E}", "E").replace(r"\mathrm", "").replace(r"\argmax", "argmax")
    clean = clean.replace(r"\mid", "|").replace("_", "").replace("^", "").replace("{", "").replace("}", "")
    surf = f.render(clean, True, color)
    math_surface_cache[cache_key] = surf
    return surf

def draw_latex(latex_str, font_size, color, x, y, center=False, right_align=False, surface=virtual_surface):
    surf = get_latex_surface(latex_str, font_size, color)
    if center:
        rect = surf.get_rect(center=(x, y))
        surface.blit(surf, rect)
    elif right_align:
        rect = surf.get_rect(topright=(x, y))
        surface.blit(surf, rect)
    else:
        surface.blit(surf, (x, y))

def update_viewport():
    global scale_factor, offset_x, offset_y
    win_w, win_h = screen.get_size()
    scale_factor = min(win_w / V_WIDTH, win_h / V_HEIGHT)
    target_w = int(V_WIDTH * scale_factor)
    target_h = int(V_HEIGHT * scale_factor)
    offset_x = (win_w - target_w) // 2
    offset_y = (win_h - target_h) // 2

def to_virtual_coords(screen_pos):
    sx, sy = screen_pos
    vx = int((sx - offset_x) / scale_factor)
    vy = int((sy - offset_y) / scale_factor)
    return vx, vy

def draw_text(text, font, color, x, y, center=False, right_align=False, surface=virtual_surface):
    img = font.render(text, True, color)
    if center:
        rect = img.get_rect(center=(x, y))
        surface.blit(img, rect)
    elif right_align:
        rect = img.get_rect(bottomright=(x, y))
        surface.blit(img, rect)
    else:
        surface.blit(img, (x, y))

def wrap_text(text, font, color, x, y, max_width, surface=virtual_surface):
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
        draw_text(line, font, color, x, y + (i * fh), surface=surface)

# ---------------------------------------------------------
# MENU SCENE
# ---------------------------------------------------------
class MenuScene:
    def __init__(self):
        self.options = [
            ("Lesson 11: MDP Foundations (S, A, T, R & Markov Property)", L11_MDPScene),
            ("Lesson 12: MDP II (Policies, Return, Value, V vs Q)", L12_PolicyScene),
            ("Lessons 4-7: Safe Corridor (Informed A* Search)", SearchScene),
            ("Lessons 13-15: Autonomous Flight (Value Iteration)", RLScene),
            ("Lessons 18-23: Stealth Tracker (Probability & HMMs)", HMMScene),
            ("Exit Command Center", None)
        ]
    
    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = to_virtual_coords(event.pos)
                for i in range(len(self.options)):
                    rect = pygame.Rect(V_WIDTH//2 - 400, 200 + i*76, 800, 58)
                    if rect.collidepoint(mx, my):
                        if self.options[i][1] is None:
                            pygame.quit()
                            sys.exit()
                        return self.options[i][1]()
        return self

    def update(self): pass

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        draw_text("USAF AI COMMAND CENTER", font_title, AIR_FORCE_YELLOW, V_WIDTH//2, 70, center=True)
        draw_text("CompSci 471: Artificial Intelligence Laboratory", font_subtitle, WHITE, V_WIDTH//2, 115, center=True)
        draw_text("Select an operational module. Maximize window for monitor scaling with LaTeX math.", font_desc, LIGHT_GRAY, V_WIDTH//2, 155, center=True)
        
        for i, (text, _) in enumerate(self.options):
            rect = pygame.Rect(V_WIDTH//2 - 400, 200 + i*76, 800, 58)
            pygame.draw.rect(surface, USAF_BLUE, rect)
            pygame.draw.rect(surface, AIR_FORCE_YELLOW, rect, 2)
            draw_text(text, font_desc, WHITE, V_WIDTH//2, 229 + i*76, center=True)

# ---------------------------------------------------------
# LESSON 11: MDP FOUNDATIONS
# ---------------------------------------------------------
class L11_MDPScene:
    def __init__(self):
        self.cols, self.rows = 4, 3
        self.cell_size = 145
        self.grid_x = 40
        self.grid_y = 160
        
        self.wall = (1, 1)
        self.goal = (3, 0)
        self.hazard = (3, 1)
        self.start_state = (0, 2)
        
        self.agent_pos = self.start_state
        self.battery = 100
        self.is_terminal = False
        self.repaired_markov = False
        
        self.last_action = None
        self.last_reward = 0
        self.cumulative_reward = 0
        self.step_history = []
        
        self.action_vectors = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}
        self.action_slips = {
            'N': [('W', 0.10), ('E', 0.10)],
            'S': [('W', 0.10), ('E', 0.10)],
            'W': [('N', 0.10), ('S', 0.10)],
            'E': [('N', 0.10), ('S', 0.10)]
        }

    def reset_env(self):
        self.agent_pos = self.start_state
        self.battery = 100
        self.is_terminal = False
        self.last_action = None
        self.last_reward = 0
        self.cumulative_reward = 0
        self.step_history.clear()

    def get_candidate_dest(self, pos, move_dir):
        dx, dy = self.action_vectors[move_dir]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) != self.wall:
            return (nx, ny)
        return pos

    def get_distribution(self, pos, action):
        if pos == self.goal or pos == self.hazard:
            return []
        intended_prob = 0.80
        slip_list = self.action_slips[action]
        outcomes = [(self.get_candidate_dest(pos, action), intended_prob, "Intended")]
        for slip_dir, p in slip_list:
            outcomes.append((self.get_candidate_dest(pos, slip_dir), p, f"Slip {slip_dir}"))
        return outcomes

    def execute_action(self, action):
        if self.is_terminal:
            return
            
        self.last_action = action
        dist = self.get_distribution(self.agent_pos, action)
        
        r_val = random.random()
        cumulative_p = 0.0
        chosen_s_prime = self.agent_pos
        chosen_desc = "Intended"
        
        for next_s, prob, desc in dist:
            cumulative_p += prob
            if r_val <= cumulative_p:
                chosen_s_prime = next_s
                chosen_desc = desc
                break

        self.battery = max(0, self.battery - 12)
        if chosen_s_prime == self.goal:
            reward = 10
            self.is_terminal = True
        elif chosen_s_prime == self.hazard:
            reward = -10
            self.is_terminal = True
        else:
            reward = -1
            
        self.last_reward = reward
        self.cumulative_reward += reward
        entry = f"s:{self.agent_pos} a:{action} -> s':{chosen_s_prime} r:{reward:+d} ({chosen_desc})"
        self.step_history.append(entry)
        if len(self.step_history) > 17:
            self.step_history.pop(0)

        self.agent_pos = chosen_s_prime

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return MenuScene()
                if event.key == pygame.K_r:
                    self.reset_env()
                if event.key == pygame.K_m:
                    self.repaired_markov = not self.repaired_markov
                    
                if not self.is_terminal:
                    if event.key == pygame.K_UP: self.execute_action('N')
                    elif event.key == pygame.K_DOWN: self.execute_action('S')
                    elif event.key == pygame.K_LEFT: self.execute_action('W')
                    elif event.key == pygame.K_RIGHT: self.execute_action('E')
        return self

    def update(self): pass

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        
        draw_text("LESSON 11: MARKOV DECISION PROCESS (MDP) FOUNDATIONS", font_title, AIR_FORCE_YELLOW, 20, 18)
        draw_text("Formulating sequential decisions under uncertainty:", font_desc, WHITE, 20, 56)
        draw_latex(r"\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle", 17, CYAN_ACCENT, 570, 53)
        draw_text("Controls: ARROWS (N/S/E/W) | 'M' = Toggle Markov Inspector | 'R' = Reset", font_info, AIR_FORCE_YELLOW, 20, 84)

        for col in range(self.cols):
            for row in range(self.rows):
                rx = self.grid_x + col * self.cell_size
                ry = self.grid_y + row * self.cell_size
                rect = pygame.Rect(rx, ry, self.cell_size, self.cell_size)
                coord_str = f"({col},{row})"
                
                if (col, row) == self.wall:
                    pygame.draw.rect(surface, WALL_COLOR, rect)
                    draw_text("WALL", font_subtitle, WHITE, rect.centerx, rect.centery, center=True)
                elif (col, row) == self.goal:
                    pygame.draw.rect(surface, (30, 100, 50), rect)
                    pygame.draw.rect(surface, GREEN, rect, 4)
                    draw_text("+10", font_title, WHITE, rect.centerx, rect.centery - 12, center=True)
                    draw_text("GOAL", font_small, LIGHT_GRAY, rect.centerx, rect.centery + 24, center=True)
                elif (col, row) == self.hazard:
                    pygame.draw.rect(surface, (110, 30, 30), rect)
                    pygame.draw.rect(surface, RED, rect, 4)
                    draw_text("-10", font_title, WHITE, rect.centerx, rect.centery - 12, center=True)
                    draw_text("HAZARD", font_small, LIGHT_GRAY, rect.centerx, rect.centery + 24, center=True)
                else:
                    pygame.draw.rect(surface, (30, 32, 42), rect)
                    pygame.draw.rect(surface, LIGHT_GRAY, rect, 1)
                    draw_text(coord_str, font_tiny, LIGHT_GRAY, rect.left + 6, rect.top + 6)
                    if (col, row) == self.start_state:
                        draw_text("START", font_small, AIR_FORCE_YELLOW, rect.centerx, rect.bottom - 20, center=True)

                if (col, row) == self.agent_pos:
                    pygame.draw.circle(surface, USAF_BLUE, rect.center, 36)
                    pygame.draw.circle(surface, AIR_FORCE_YELLOW, rect.center, 36, 4)
                    pygame.draw.circle(surface, WHITE, rect.center, 12)

        # Markov Inspector Box with LaTeX Property
        insp_y = self.grid_y + (self.rows * self.cell_size) + 20
        insp_w = self.cols * self.cell_size
        pygame.draw.rect(surface, PANEL_GRAY, (self.grid_x, insp_y, insp_w, 230))
        pygame.draw.rect(surface, AIR_FORCE_YELLOW if self.repaired_markov else RED, (self.grid_x, insp_y, insp_w, 230), 2)
        
        mode_title = "MARKOV PROPERTY: [REPAIRED STATE]" if self.repaired_markov else "MARKOV PROPERTY: [INCOMPLETE STATE]"
        mode_col = GREEN if self.repaired_markov else RED
        draw_text(mode_title, font_subtitle, mode_col, self.grid_x + 15, insp_y + 15)
        
        if not self.repaired_markov:
            draw_latex(r"s = (x, y) \quad \text{violates} \quad P(S_{t+1}, R_{t+1} \mid S_t, A_t, \mathrm{history}) = P(S_{t+1}, R_{t+1} \mid S_t, A_t)", 16, WHITE, self.grid_x + 15, insp_y + 46)
            wrap_text("Limitation: State only encodes coordinate. If unobserved variables (e.g. battery drain) alter transition outcomes, past history is required to predict the future. The Markov test fails!", font_small, LIGHT_GRAY, self.grid_x + 15, insp_y + 82, insp_w - 30)
            draw_text(f"Hidden Battery: {self.battery}% (Unobserved by agent)", font_desc, AIR_FORCE_YELLOW, self.grid_x + 15, insp_y + 180)
        else:
            draw_latex(r"s = (x, y, \mathrm{battery}) \implies P(S_{t+1}, R_{t+1} \mid S_t, A_t, \mathrm{history}) = P(S_{t+1}, R_{t+1} \mid S_t, A_t)", 16, GREEN, self.grid_x + 15, insp_y + 46)
            wrap_text("Markov Property Satisfied: All relevant historical effects are encapsulated in the current state snapshot. The next state distribution depends strictly on the current state and action.", font_small, LIGHT_GRAY, self.grid_x + 15, insp_y + 82, insp_w - 30)
            draw_text(f"Observed Battery: {self.battery}% (Encoded into state vector)", font_desc, GREEN, self.grid_x + 15, insp_y + 180)

        panel_x = self.grid_x + insp_w + 30
        panel_y = self.grid_y
        panel_w = V_WIDTH - panel_x - 30
        panel_h = V_HEIGHT - panel_y - 45
        
        pygame.draw.rect(surface, PANEL_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(surface, LIGHT_GRAY, (panel_x, panel_y, panel_w, panel_h), 2)
        
        draw_text("MDP SYSTEM TELEMETRY", font_subtitle, AIR_FORCE_YELLOW, panel_x + 15, panel_y + 15)
        draw_text(f"Current State (s): {self.agent_pos}", font_info, WHITE, panel_x + 15, panel_y + 48)
        draw_text(f"Immediate Reward (r): {self.last_reward:+d}", font_info, GREEN if self.last_reward >= 0 else RED, panel_x + 15, panel_y + 75)
        draw_text(f"Cumulative Score: {self.cumulative_reward:+d}", font_info, WHITE, panel_x + 320, panel_y + 75)
        
        pygame.draw.line(surface, LIGHT_GRAY, (panel_x + 15, panel_y + 105), (panel_x + panel_w - 15, panel_y + 105))
        draw_text("TRANSITION DISTRIBUTION MODEL:", font_info, AIR_FORCE_YELLOW, panel_x + 15, panel_y + 115)
        draw_latex(r"P(s' \mid s, a)", 17, CYAN_ACCENT, panel_x + 335, panel_y + 112)
        
        if self.last_action and not self.is_terminal:
            dist = self.get_distribution(self.agent_pos, self.last_action)
            draw_text(f"Action Evaluated: {self.last_action}", font_desc, WHITE, panel_x + 15, panel_y + 140)
            
            pygame.draw.rect(surface, USAF_BLUE, (panel_x + 15, panel_y + 168, panel_w - 30, 26))
            draw_text("Candidate Next State s'", font_tiny, WHITE, panel_x + 25, panel_y + 174)
            draw_latex(r"P(s' \mid s, a)", 13, WHITE, panel_x + 250, panel_y + 171)
            draw_text("Outcome Type", font_tiny, WHITE, panel_x + 370, panel_y + 174)
            draw_text("Reward", font_tiny, WHITE, panel_x + 530, panel_y + 174)
            
            for i, (next_s, p, o_type) in enumerate(dist):
                row_y = panel_y + 200 + (i * 24)
                r_est = 10 if next_s == self.goal else (-10 if next_s == self.hazard else -1)
                draw_text(f"{next_s}", font_small, WHITE, panel_x + 25, row_y)
                draw_text(f"{p:.2f} ({int(p*100)}%)", font_small, AIR_FORCE_YELLOW, panel_x + 250, row_y)
                draw_text(o_type, font_small, LIGHT_GRAY, panel_x + 370, row_y)
                draw_text(f"{r_est:+d}", font_small, GREEN if r_est > 0 else RED, panel_x + 540, row_y)
        else:
            if self.is_terminal:
                draw_text("TERMINAL STATE REACHED. Press 'R' to restart.", font_desc, GREEN if self.agent_pos == self.goal else RED, panel_x + 15, panel_y + 145)
            else:
                draw_text("Select an action (ARROW KEYS) to evaluate transitions.", font_desc, LIGHT_GRAY, panel_x + 15, panel_y + 145)

        pygame.draw.line(surface, LIGHT_GRAY, (panel_x + 15, panel_y + 290), (panel_x + panel_w - 15, panel_y + 290))
        draw_text("EXPERIENCE DISPATCH LOG (s -> a -> s', r)", font_info, AIR_FORCE_YELLOW, panel_x + 15, panel_y + 302)
        
        for i, log_entry in enumerate(reversed(self.step_history)):
            draw_text(log_entry, font_tiny, WHITE if i == 0 else LIGHT_GRAY, panel_x + 15, panel_y + 330 + (i * 21))

        draw_text("Press ESC to return to Menu", font_desc, AIR_FORCE_YELLOW, 20, V_HEIGHT - 30)

# ---------------------------------------------------------
# LESSON 12: MDP II - POLICIES, RETURN, AND VALUE FUNCTIONS
# ---------------------------------------------------------
class L12_PolicyScene:
    def __init__(self):
        self.cols, self.rows = 4, 3
        self.cell_size = 145
        self.grid_x = 40
        self.grid_y = 160
        
        self.wall = (1, 1)
        self.goal = (3, 0)
        self.hazard = (3, 1)
        self.start_state = (0, 2)
        
        self.policy_names = [
            "Policy 1 (Safe Detour Around Hazard)",
            "Policy 2 (Risky / Direct Cliff Walk)",
            "Policy 3 (Sub-optimal / Defective Loop)"
        ]
        self.policies = [
            {
                (0,0): 'E', (1,0): 'E', (2,0): 'E',
                (0,1): 'N',             (2,1): 'N',
                (0,2): 'N', (1,2): 'W', (2,2): 'N', (3,2): 'W'
            },
            {
                (0,0): 'E', (1,0): 'E', (2,0): 'E',
                (0,1): 'N',             (2,1): 'N',
                (0,2): 'E', (1,2): 'E', (2,2): 'E', (3,2): 'N'
            },
            {
                (0,0): 'S', (1,0): 'W', (2,0): 'W',
                (0,1): 'S',             (2,1): 'S',
                (0,2): 'E', (1,2): 'E', (2,2): 'S', (3,2): 'W'
            }
        ]
        self.policy_idx = 0
        self.gamma = 0.90
        self.view_mode = "V"
        self.selected_cell = (0, 2)
        
        self.agent_pos = self.start_state
        self.is_terminal = False
        self.trajectory_log = []
        self.sim_return = 0.0
        self.step_k = 0
        
        self.action_vectors = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}
        self.action_slips = {
            'N': [('W', 0.10), ('E', 0.10)],
            'S': [('W', 0.10), ('E', 0.10)],
            'W': [('N', 0.10), ('S', 0.10)],
            'E': [('N', 0.10), ('S', 0.10)]
        }
        
        self.V = {}
        self.Q = {}
        self.recompute_values()

    def get_candidate_dest(self, pos, move_dir):
        dx, dy = self.action_vectors[move_dir]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) != self.wall:
            return (nx, ny)
        return pos

    def get_distribution(self, pos, action):
        if pos == self.goal or pos == self.hazard:
            return []
        outcomes = [(self.get_candidate_dest(pos, action), 0.80)]
        for s_dir, p in self.action_slips[action]:
            outcomes.append((self.get_candidate_dest(pos, s_dir), p))
        return outcomes

    def recompute_values(self):
        current_policy = self.policies[self.policy_idx]
        all_states = [(c, r) for c in range(self.cols) for r in range(self.rows) if (c, r) != self.wall]
        self.V = {s: 0.0 for s in all_states}
        
        for _ in range(120):
            new_V = {}
            for s in all_states:
                if s == self.goal or s == self.hazard:
                    new_V[s] = 0.0
                    continue
                action = current_policy[s]
                expected_v = 0.0
                for s_prime, p in self.get_distribution(s, action):
                    r = 10.0 if s_prime == self.goal else (-10.0 if s_prime == self.hazard else -1.0)
                    expected_v += p * (r + self.gamma * self.V[s_prime])
                new_V[s] = expected_v
            self.V = new_V
            
        self.Q = {}
        for s in all_states:
            if s == self.goal or s == self.hazard:
                continue
            self.Q[s] = {}
            for a in ['N', 'S', 'W', 'E']:
                q_val = 0.0
                for s_prime, p in self.get_distribution(s, a):
                    r = 10.0 if s_prime == self.goal else (-10.0 if s_prime == self.hazard else -1.0)
                    q_val += p * (r + self.gamma * self.V[s_prime])
                self.Q[s][a] = q_val

    def step_trajectory(self):
        if self.is_terminal:
            return
            
        curr_policy = self.policies[self.policy_idx]
        chosen_action = curr_policy[self.agent_pos]
        dist = self.get_distribution(self.agent_pos, chosen_action)
        
        r_val = random.random()
        cum_p = 0.0
        chosen_next = self.agent_pos
        for next_s, p in dist:
            cum_p += p
            if r_val <= cum_p:
                chosen_next = next_s
                break
                
        reward = 10.0 if chosen_next == self.goal else (-10.0 if chosen_next == self.hazard else -1.0)
        if chosen_next in [self.goal, self.hazard]:
            self.is_terminal = True
            
        discounted_r = (self.gamma ** self.step_k) * reward
        self.sim_return += discounted_r
        
        log_str = f"t={self.step_k} s:{self.agent_pos} a:{chosen_action}->s':{chosen_next} r:{reward:+1.0f} (g^{self.step_k}*r={discounted_r:+.2f})"
        self.trajectory_log.append(log_str)
        if len(self.trajectory_log) > 13:
            self.trajectory_log.pop(0)
            
        self.agent_pos = chosen_next
        self.step_k += 1

    def reset_trajectory(self):
        self.agent_pos = self.start_state
        self.is_terminal = False
        self.trajectory_log.clear()
        self.sim_return = 0.0
        self.step_k = 0

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return MenuScene()
                if event.key == pygame.K_p:
                    self.policy_idx = (self.policy_idx + 1) % len(self.policies)
                    self.recompute_values()
                    self.reset_trajectory()
                if event.key == pygame.K_v:
                    self.view_mode = "Q" if self.view_mode == "V" else "V"
                if event.key == pygame.K_SPACE:
                    self.step_trajectory()
                if event.key == pygame.K_r:
                    self.reset_trajectory()
                if event.key == pygame.K_LEFTBRACKET:
                    self.gamma = max(0.0, round(self.gamma - 0.05, 2))
                    self.recompute_values()
                if event.key == pygame.K_RIGHTBRACKET:
                    self.gamma = min(0.99, round(self.gamma + 0.05, 2))
                    self.recompute_values()
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = to_virtual_coords(event.pos)
                col = (mx - self.grid_x) // self.cell_size
                row = (my - self.grid_y) // self.cell_size
                if 0 <= col < self.cols and 0 <= row < self.rows:
                    if (col, row) != self.wall and (col, row) != self.goal and (col, row) != self.hazard:
                        self.selected_cell = (col, row)
        return self

    def update(self): pass

    def draw_policy_arrow(self, surface, color, cx, cy, direction):
        length = 26
        if direction == 'N': s, e = (cx, cy + length), (cx, cy - length)
        elif direction == 'S': s, e = (cx, cy - length), (cx, cy + length)
        elif direction == 'W': s, e = (cx + length, cy), (cx - length, cy)
        elif direction == 'E': s, e = (cx - length, cy), (cx + length, cy)
        else: return
        pygame.draw.line(surface, color, s, e, 4)
        pygame.draw.circle(surface, color, e, 6)

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        
        draw_text("LESSON 12: MDP POLICIES, RETURN, AND VALUE FUNCTIONS", font_title, AIR_FORCE_YELLOW, 20, 18)
        draw_text("Evaluating long-term behavior with discount factor:", font_desc, WHITE, 20, 56)
        draw_latex(r"G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \quad \text{and} \quad V^{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]", 17, CYAN_ACCENT, 525, 52)
        draw_text("Controls: 'P' = Cycle Policy | 'V' = Toggle V/Q | '['/']' = Gamma | SPACE = Step Path | 'R' = Reset", font_info, AIR_FORCE_YELLOW, 20, 84)

        current_policy = self.policies[self.policy_idx]

        for col in range(self.cols):
            for row in range(self.rows):
                rx = self.grid_x + col * self.cell_size
                ry = self.grid_y + row * self.cell_size
                rect = pygame.Rect(rx, ry, self.cell_size, self.cell_size)
                
                if (col, row) == self.wall:
                    pygame.draw.rect(surface, WALL_COLOR, rect)
                    draw_text("WALL", font_subtitle, WHITE, rect.centerx, rect.centery, center=True)
                elif (col, row) == self.goal:
                    pygame.draw.rect(surface, (30, 100, 50), rect)
                    pygame.draw.rect(surface, GREEN, rect, 4)
                    draw_text("+10", font_title, WHITE, rect.centerx, rect.centery - 12, center=True)
                    draw_text("TERMINAL", font_small, LIGHT_GRAY, rect.centerx, rect.centery + 24, center=True)
                elif (col, row) == self.hazard:
                    pygame.draw.rect(surface, (110, 30, 30), rect)
                    pygame.draw.rect(surface, RED, rect, 4)
                    draw_text("-10", font_title, WHITE, rect.centerx, rect.centery - 12, center=True)
                    draw_text("TERMINAL", font_small, LIGHT_GRAY, rect.centerx, rect.centery + 24, center=True)
                else:
                    bg_col = (45, 48, 65) if (col, row) == self.selected_cell else (30, 32, 42)
                    pygame.draw.rect(surface, bg_col, rect)
                    border_col = CYAN_ACCENT if (col, row) == self.selected_cell else LIGHT_GRAY
                    pygame.draw.rect(surface, border_col, rect, 2 if (col, row) == self.selected_cell else 1)
                    draw_text(f"({col},{row})", font_tiny, LIGHT_GRAY, rect.left + 6, rect.top + 6)
                    
                    if self.view_mode == "V":
                        v_val = self.V.get((col, row), 0.0)
                        val_col = GREEN if v_val > 0 else (RED if v_val < -1 else WHITE)
                        draw_latex(f"V^{{\\pi}}={v_val:+.2f}", 16, val_col, rect.centerx, rect.centery - 24, center=True)
                        
                        p_dir = current_policy.get((col, row))
                        self.draw_policy_arrow(surface, AIR_FORCE_YELLOW, rect.centerx, rect.centery + 20, p_dir)
                    else:
                        q_dict = self.Q.get((col, row), {})
                        if q_dict:
                            draw_latex(f"N:{q_dict['N']:+.1f}", 12, WHITE, rect.centerx, rect.top + 18, center=True)
                            draw_latex(f"S:{q_dict['S']:+.1f}", 12, WHITE, rect.centerx, rect.bottom - 18, center=True)
                            draw_latex(f"W:{q_dict['W']:+.1f}", 12, WHITE, rect.left + 26, rect.centery, center=True)
                            draw_latex(f"E:{q_dict['E']:+.1f}", 12, WHITE, rect.right - 26, rect.centery, center=True)
                            p_dir = current_policy.get((col, row))
                            draw_text(f"pi:{p_dir}", font_tiny, AIR_FORCE_YELLOW, rect.centerx, rect.centery, center=True)

                if (col, row) == self.agent_pos:
                    pygame.draw.circle(surface, USAF_BLUE, rect.center, 34)
                    pygame.draw.circle(surface, AIR_FORCE_YELLOW, rect.center, 34, 4)
                    pygame.draw.circle(surface, WHITE, rect.center, 10)

        # Theoretical Foundation Box with Formatted LaTeX
        insp_y = self.grid_y + (self.rows * self.cell_size) + 20
        insp_w = self.cols * self.cell_size
        pygame.draw.rect(surface, PANEL_GRAY, (self.grid_x, insp_y, insp_w, 230))
        pygame.draw.rect(surface, AIR_FORCE_YELLOW, (self.grid_x, insp_y, insp_w, 230), 2)
        
        draw_text("THEORETICAL FOUNDATION: POLICY EVALUATION & VALUES", font_subtitle, AIR_FORCE_YELLOW, self.grid_x + 15, insp_y + 14)
        draw_latex(r"r_t \longrightarrow G_t \longrightarrow V^\pi(s) \longrightarrow Q^\pi(s, a) \longrightarrow \pi^*(s) = \argmax_a Q^\pi(s, a)", 16, CYAN_ACCENT, self.grid_x + 15, insp_y + 42)
        
        draw_latex(r"\bullet \; V^\pi(s) = \sum_{s'} P(s' \mid s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma V^\pi(s') \right] \quad (\text{Value of BEING in } s)", 15, WHITE, self.grid_x + 15, insp_y + 78)
        draw_latex(r"\bullet \; Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right] \quad (\text{Value of FORCING } a \text{ then following } \pi)", 15, WHITE, self.grid_x + 15, insp_y + 118)
        wrap_text(f"Discount Factor (gamma = {self.gamma:.2f}): Governs the horizon. Low gamma values immediate rewards; high gamma considers long-term downstream consequences.", font_small, LIGHT_GRAY, self.grid_x + 15, insp_y + 162, insp_w - 30)

        panel_x = self.grid_x + insp_w + 30
        panel_y = self.grid_y
        panel_w = V_WIDTH - panel_x - 30
        panel_h = V_HEIGHT - panel_y - 45
        
        pygame.draw.rect(surface, PANEL_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(surface, LIGHT_GRAY, (panel_x, panel_y, panel_w, panel_h), 2)
        
        draw_text("POLICY & ACTION-VALUE TELEMETRY", font_subtitle, AIR_FORCE_YELLOW, panel_x + 15, panel_y + 15)
        draw_text(f"Active: {self.policy_names[self.policy_idx]}", font_info, WHITE, panel_x + 15, panel_y + 46)
        draw_latex(f"\\gamma = {self.gamma:.2f}", 16, CYAN_ACCENT, panel_x + 15, panel_y + 72)
        draw_text(f"Display Mode: {'State Value V(s)' if self.view_mode=='V' else 'Action Value Q(s,a)'}", font_info, AIR_FORCE_YELLOW, panel_x + 280, panel_y + 72)
        
        pygame.draw.line(surface, LIGHT_GRAY, (panel_x + 15, panel_y + 100), (panel_x + panel_w - 15, panel_y + 100))
        sc = self.selected_cell
        draw_text(f"INSPECTED STATE s = {sc} (Click any grid cell)", font_subtitle, CYAN_ACCENT, panel_x + 15, panel_y + 110)
        
        q_sc = self.Q.get(sc, {})
        v_sc = self.V.get(sc, 0.0)
        p_act = current_policy.get(sc, 'None')
        draw_latex(f"V^{{\\pi}}({sc}) = {v_sc:+.3f} \\quad \\pi({sc}) = \\text{{{p_act}}}", 16, WHITE, panel_x + 15, panel_y + 138)
        
        if q_sc:
            best_a = max(q_sc, key=q_sc.get)
            draw_latex(f"Q(s, N) = {q_sc['N']:+.3f} \\quad Q(s, S) = {q_sc['S']:+.3f}", 15, LIGHT_GRAY, panel_x + 15, panel_y + 166)
            draw_latex(f"Q(s, W) = {q_sc['W']:+.3f} \\quad Q(s, E) = {q_sc['E']:+.3f}", 15, LIGHT_GRAY, panel_x + 15, panel_y + 192)
            draw_latex(f"\\argmax_a Q(s, a) = \\text{{{best_a}}} \\quad (Q^* = {q_sc[best_a]:+.3f})", 16, GREEN, panel_x + 15, panel_y + 220)
            if best_a != p_act:
                draw_text("Sub-optimal choice: Active policy does not match greedy Q-action!", font_small, RED, panel_x + 15, panel_y + 248)
            else:
                draw_text("Optimal choice: Active policy matches argmax_a Q(s,a).", font_small, GREEN, panel_x + 15, panel_y + 248)

        pygame.draw.line(surface, LIGHT_GRAY, (panel_x + 15, panel_y + 276), (panel_x + panel_w - 15, panel_y + 276))
        draw_text("TRAJECTORY ROLLOUT SIMULATOR (SPACEBAR)", font_subtitle, AIR_FORCE_YELLOW, panel_x + 15, panel_y + 288)
        draw_latex(f"G_0 = {self.sim_return:+.2f} \\quad \\text{{vs}} \\quad V^{{\\pi}}(s_0) = {self.V.get(self.start_state, 0.0):+.2f}", 16, WHITE, panel_x + 15, panel_y + 314)
        wrap_text("A single trajectory return G_0 reflects sampled uncertainty; the state value V(s) is the mathematical expectation over all possible stochastic paths.", font_small, LIGHT_GRAY, panel_x + 15, panel_y + 342, panel_w - 30)

        for i, log_entry in enumerate(reversed(self.trajectory_log)):
            draw_text(log_entry, font_tiny, WHITE if i == 0 else LIGHT_GRAY, panel_x + 15, panel_y + 395 + (i * 20))

        draw_text("Press ESC to return to Menu", font_desc, AIR_FORCE_YELLOW, 20, V_HEIGHT - 30)

# ---------------------------------------------------------
# MODULE 1: SEARCH (Lessons 4-7)
# ---------------------------------------------------------
class SearchScene:
    def __init__(self):
        self.cols, self.rows = 12, 10
        self.cell_size = 68
        self.grid_offset_x = 30
        self.grid_offset_y = 150
        self.obstacles = set()
        self.start = (0, 0)
        self.goal = (11, 9)
        self.desc = "LESSONS 4-7: Informed Search. Evaluates nodes using cost and heuristic distance. RIGHT ARROW = Step-by-Step. SPACEBAR = Auto-Search."
        self.reset_search()

    def reset_search(self):
        self.state = "EDIT"
        self.path = []
        self.open_set = []
        self.explored = set()
        self.g_score = {self.start: 0}
        self.f_score = {self.start: self.heuristic(self.start, self.goal)}
        self.came_from = {}
        self.current_eval = None
        self.eval_history = []
        self.step_timer = 0
        self.auto_delay = 30 
        self.avatar_delay = 180 
        self.avatar_index = 0

    def init_search(self):
        heapq.heappush(self.open_set, (self.f_score[self.start], id(self.start), self.start))

    def get_neighbors(self, node):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in self.obstacles]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step_astar(self):
        if not self.open_set:
            self.state = "NO_PATH"
            return
            
        _, _, current = heapq.heappop(self.open_set)
        if current in self.explored: 
            return
            
        self.explored.add(current)
        g = self.g_score[current]
        h = self.heuristic(current, self.goal)
        f = g + h
        self.current_eval = {'node': current, 'g': g, 'h': h, 'f': f}
        
        self.eval_history.append(f"Eval {current}: f={f} (g={g}, h={h})")
        if len(self.eval_history) > 23:
            self.eval_history.pop(0)
        
        if current == self.goal:
            self.reconstruct_path()
            return
            
        for neighbor in self.get_neighbors(current):
            if neighbor in self.explored:
                continue
            tentative_g = self.g_score[current] + 1
            if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                new_f = tentative_g + self.heuristic(neighbor, self.goal)
                self.f_score[neighbor] = new_f
                heapq.heappush(self.open_set, (new_f, id(neighbor), neighbor))

    def reconstruct_path(self):
        curr = self.goal
        while curr != self.start:
            self.path.append(curr)
            curr = self.came_from[curr]
        self.path.append(self.start)
        self.path.reverse()
        self.state = "PATH_FOUND"
        self.avatar_index = 0
        
        self.eval_history = []
        for i, node in enumerate(self.path):
            g = self.g_score[node]
            h = self.heuristic(node, self.goal)
            self.eval_history.append(f"Step {i:02d} {node}: f({g+h}) = g({g}) + h({h})")

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return MenuScene()
                if event.key == pygame.K_RIGHT:
                    if self.state == "EDIT":
                        self.state = "STEPPING"
                        self.init_search()
                    if self.state in ["STEPPING", "AUTO"]:
                        self.state = "STEPPING"
                        self.step_astar()
                if event.key == pygame.K_SPACE:
                    if self.state in ["PATH_FOUND", "NO_PATH"]:
                        self.reset_search()
                    else:
                        if self.state == "EDIT":
                            self.init_search()
                        self.state = "AUTO"
                        
            if event.type == pygame.MOUSEBUTTONDOWN and self.state in ["EDIT", "PATH_FOUND", "NO_PATH"]:
                if self.state != "EDIT":
                    self.reset_search()
                mx, my = to_virtual_coords(event.pos)
                gx = (mx - self.grid_offset_x) // self.cell_size
                gy = (my - self.grid_offset_y) // self.cell_size
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    if (gx, gy) != self.start and (gx, gy) != self.goal:
                        if (gx, gy) in self.obstacles:
                            self.obstacles.remove((gx, gy))
                        else:
                            self.obstacles.add((gx, gy))
        return self

    def update(self):
        now = pygame.time.get_ticks()
        if self.state == "AUTO":
            if now - self.step_timer > self.auto_delay:
                self.step_timer = now
                self.step_astar()
        elif self.state == "PATH_FOUND":
            if now - self.step_timer > self.avatar_delay:
                self.step_timer = now
                if self.avatar_index < len(self.path) - 1:
                    self.avatar_index += 1

    def draw_uav(self, surface, cx, cy, dx, dy):
        base_points = [(22, 0), (-16, -16), (-6, 0), (-16, 16)]
        if dx == 1:   pts = [(cx + x, cy + y) for x, y in base_points]
        elif dx == -1: pts = [(cx - x, cy - y) for x, y in base_points]
        elif dy == 1:  pts = [(cx - y, cy + x) for x, y in base_points]
        elif dy == -1: pts = [(cx + y, cy - x) for x, y in base_points]
        else: pts = [(cx + x, cy + y) for x, y in base_points]
        pygame.draw.polygon(surface, AIR_FORCE_YELLOW, pts)

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        draw_text("OPERATION: SAFE CORRIDOR (A* Search Visualization)", font_title, AIR_FORCE_YELLOW, 20, 20)
        wrap_text(self.desc, font_desc, WHITE, 20, 65, V_WIDTH - 40)
        draw_latex(r"f(n) = g(n) + h(n) \quad \text{where} \quad h(n) = |x_n - x_{\mathrm{goal}}| + |y_n - y_{\mathrm{goal}}|", 17, CYAN_ACCENT, 20, 98)

        for x in range(self.cols):
            for y in range(self.rows):
                rect = pygame.Rect(self.grid_offset_x + x*self.cell_size, self.grid_offset_y + y*self.cell_size, self.cell_size, self.cell_size)
                if (x, y) == self.start:
                    pygame.draw.rect(surface, USAF_BLUE, rect)
                elif (x, y) == self.goal:
                    pygame.draw.rect(surface, GREEN, rect)
                elif (x, y) in self.obstacles:
                    pygame.draw.rect(surface, RED, rect)
                elif self.state == "PATH_FOUND" and (x, y) in self.path[:self.avatar_index+1]:
                    pygame.draw.rect(surface, AIR_FORCE_YELLOW, rect, 4)
                elif (x, y) in self.explored:
                    pygame.draw.rect(surface, EXPLORED_COLOR, rect)
                elif any((x, y) == item[2] for item in self.open_set):
                    pygame.draw.rect(surface, FRONTIER_COLOR, rect)
                else:
                    pygame.draw.rect(surface, LIGHT_GRAY, rect, 1)

                if (x, y) in self.g_score and (x, y) not in self.obstacles:
                    g = self.g_score[(x,y)]
                    h = self.heuristic((x,y), self.goal)
                    f = g + h
                    if (x, y) == self.start:
                        draw_text("HQ", font_info, WHITE, rect.centerx, rect.centery, center=True)
                    elif (x, y) == self.goal:
                        draw_text("OBJ", font_info, DARK_GRAY, rect.centerx, rect.centery, center=True)
                    else:
                        draw_text(f"f:{f}", font_tiny, GREEN, rect.centerx, rect.top + 5, center=True)
                        draw_text(f"g:{g}", font_tiny, WHITE, rect.left + 4, rect.bottom - 16)
                        draw_text(f"h:{h}", font_tiny, AIR_FORCE_YELLOW, rect.right - 4, rect.bottom - 4, right_align=True)

                if (x, y) in self.obstacles:
                    draw_text("SAM", font_small, WHITE, rect.centerx, rect.centery, center=True)

        if self.state == "PATH_FOUND" and len(self.path) > 0:
            curr_pos = self.path[self.avatar_index]
            cx = self.grid_offset_x + curr_pos[0] * self.cell_size + self.cell_size // 2
            cy = self.grid_offset_y + curr_pos[1] * self.cell_size + self.cell_size // 2
            dx, dy = 1, 0 
            if self.avatar_index > 0:
                prev_pos = self.path[self.avatar_index - 1]
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
            self.draw_uav(surface, cx, cy, dx, dy)

        panel_x = self.grid_offset_x + (self.cols * self.cell_size) + 30
        panel_y = self.grid_offset_y
        panel_w = V_WIDTH - panel_x - 30
        panel_h = self.rows * self.cell_size
        pygame.draw.rect(surface, PANEL_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(surface, LIGHT_GRAY, (panel_x, panel_y, panel_w, panel_h), 2)
        
        if self.state == "EDIT":
            draw_text("SYSTEM IDLE", font_info, WHITE, panel_x + 10, panel_y + 10)
            wrap_text("A* Path Cost Mechanics:\n\nf remains constant along the shortest route because g (+1) and h (-1) balance out. Place SAM sites to force detours.", font_small, LIGHT_GRAY, panel_x + 10, panel_y + 50, panel_w - 20)
        elif self.state in ["STEPPING", "AUTO"]:
            draw_text("SEARCH HISTORY LOG", font_info, AIR_FORCE_YELLOW, panel_x + 10, panel_y + 10)
            if self.current_eval:
                ce = self.current_eval
                draw_text(f"Node: {ce['node']}", font_info, WHITE, panel_x + 10, panel_y + 40)
                draw_latex(f"g(n) = {ce['g']} \\quad h(n) = {ce['h']}", 15, WHITE, panel_x + 10, panel_y + 68)
                draw_latex(f"f(n) = g(n) + h(n) = {ce['f']}", 16, GREEN, panel_x + 10, panel_y + 96)
            pygame.draw.line(surface, LIGHT_GRAY, (panel_x+10, panel_y+130), (panel_x+panel_w-10, panel_y+130))
            for i, log_str in enumerate(reversed(self.eval_history)):
                draw_text(log_str, font_tiny, WHITE if i == 0 else LIGHT_GRAY, panel_x + 10, panel_y + 140 + i*18)
        elif self.state == "PATH_FOUND":
            draw_text("FINAL PATH TABLE", font_info, GREEN, panel_x + 10, panel_y + 10)
            draw_text(f"Optimal Steps: {len(self.path)-1}", font_small, WHITE, panel_x + 10, panel_y + 35)
            pygame.draw.line(surface, LIGHT_GRAY, (panel_x+10, panel_y+60), (panel_x+panel_w-10, panel_y+60))
            for i, log_str in enumerate(self.eval_history):
                color = AIR_FORCE_YELLOW if i == self.avatar_index else LIGHT_GRAY
                draw_text(log_str, font_small, color, panel_x + 10, panel_y + 70 + i*22)
        elif self.state == "NO_PATH":
            draw_text("MISSION FAILED: NO VALID PATH", font_info, RED, panel_x + 10, panel_y + 10)

        draw_text("Press ESC to return to Menu", font_desc, AIR_FORCE_YELLOW, 20, V_HEIGHT - 30)

# ---------------------------------------------------------
# MODULE 2: VALUE ITERATION (Lessons 13-15)
# ---------------------------------------------------------
class RLScene:
    def __init__(self):
        self.size = 6
        self.cell_size = 95
        self.grid_offset_x = (V_WIDTH - (self.size * self.cell_size)) // 2
        self.grid_offset_y = 200
        self.values = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        self.policy = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.goal = (5, 5)
        self.hazards = {(2, 2): -100, (2, 3): -100, (4, 1): -50} 
        self.gamma = 0.9 
        self.desc = "LESSONS 13-15: MDP Value Iteration. Bellman updates propagate future cumulative expected return backwards. Press SPACEBAR to iterate."

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
                        v = 0 + self.gamma * self.values[ny][nx]
                    else:
                        v = -1 + self.gamma * self.values[y][x]
                    
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
        length = 32
        if direction == 'U': start, end = (x, y+length), (x, y-length)
        elif direction == 'D': start, end = (x, y-length), (x, y+length)
        elif direction == 'L': start, end = (x+length, y), (x-length, y)
        elif direction == 'R': start, end = (x-length, y), (x+length, y)
        else: return
        pygame.draw.line(surface, color, start, end, 4)
        pygame.draw.circle(surface, color, end, 6)

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        draw_text("OPERATION: AUTONOMOUS FLIGHT (Value Iteration)", font_title, AIR_FORCE_YELLOW, 20, 20)
        wrap_text(self.desc, font_desc, WHITE, 20, 65, V_WIDTH - 40)
        draw_latex(r"V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]", 17, CYAN_ACCENT, 20, 98)
        draw_text("Press SPACE to perform Bellman Update. ESC to return.", font_desc, AIR_FORCE_YELLOW, 20, V_HEIGHT - 40)

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(self.grid_offset_x + x*self.cell_size, self.grid_offset_y + y*self.cell_size, self.cell_size, self.cell_size)
                if (x, y) == self.goal:
                    pygame.draw.rect(surface, GREEN, rect)
                    draw_text("+100", font_info, DARK_GRAY, rect.centerx, rect.centery, center=True)
                elif (x, y) in self.hazards:
                    pygame.draw.rect(surface, RED, rect)
                    draw_text(str(self.hazards[(x,y)]), font_info, WHITE, rect.centerx, rect.centery, center=True)
                else:
                    val_color = min(255, max(0, int((self.values[y][x] / 100.0) * 255)))
                    pygame.draw.rect(surface, (val_color, val_color, val_color), rect)
                    pygame.draw.rect(surface, LIGHT_GRAY, rect, 1)
                    draw_text(f"{self.values[y][x]:.1f}", font_small, AIR_FORCE_YELLOW, rect.left + 5, rect.top + 5)
                    if self.policy[y][x]:
                        self.draw_arrow(surface, USAF_BLUE, rect.centerx, rect.centery, self.policy[y][x])

# ---------------------------------------------------------
# MODULE 3: HMM & PROBABILITY (Lessons 18-23)
# ---------------------------------------------------------
class HMMScene:
    def __init__(self):
        self.num_sectors = 10
        self.true_pos = 0 
        self.belief = [1.0 / self.num_sectors for _ in range(self.num_sectors)]
        self.desc = "LESSONS 18-23: Probability & HMMs. Inferring true coordinates from noisy sensors using Bayes' Rule and Markov transitions."

    def update_hmm(self):
        if random.random() < 0.8 and self.true_pos < self.num_sectors - 1:
            self.true_pos += 1
            
        new_belief = [0.0] * self.num_sectors
        for i in range(self.num_sectors):
            new_belief[i] += self.belief[i] * 0.2
            if i > 0:
                new_belief[i] += self.belief[i-1] * 0.8
        self.belief = new_belief

        obs = self.true_pos
        r = random.random()
        if r < 0.2: obs = max(0, self.true_pos - 1)
        elif r > 0.8: obs = min(self.num_sectors - 1, self.true_pos + 1)

        for i in range(self.num_sectors):
            if i == obs: p_z = 0.6
            elif abs(i - obs) == 1: p_z = 0.2
            else: p_z = 0.01 
            self.belief[i] = self.belief[i] * p_z

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

    def draw(self, surface):
        surface.fill(DARK_GRAY)
        draw_text("OPERATION: STEALTH TRACKER (Hidden Markov Models)", font_title, AIR_FORCE_YELLOW, 20, 20)
        wrap_text(self.desc, font_desc, WHITE, 20, 65, V_WIDTH - 40)
        draw_latex(r"P(S_t \mid z_{1:t}) \propto P(z_t \mid S_t) \sum_{S_{t-1}} P(S_t \mid S_{t-1}) P(S_{t-1} \mid z_{1:t-1})", 17, CYAN_ACCENT, 20, 98)
        draw_text("Press SPACEBAR to advance 1 time step. ESC to return to Menu.", font_desc, AIR_FORCE_YELLOW, 20, V_HEIGHT - 40)

        bar_width = 95
        start_x = (V_WIDTH - (self.num_sectors * (bar_width + 10))) // 2
        
        for i in range(self.num_sectors):
            x = start_x + i * (bar_width + 10)
            prob = self.belief[i]
            bar_height = int(prob * 450)
            rect = pygame.Rect(x, 650 - bar_height, bar_width, bar_height)
            pygame.draw.rect(surface, RADAR_GREEN, rect)
            
            draw_text(f"Sec {i}", font_desc, WHITE, x + bar_width//2, 670, center=True)
            draw_text(f"{prob*100:.1f}%", font_small, AIR_FORCE_YELLOW, x + bar_width//2, 630 - bar_height, center=True)

            if i == self.true_pos:
                plane_rect = pygame.Rect(x + 15, 730, 65, 26)
                pygame.draw.rect(surface, USAF_BLUE, plane_rect)
                draw_text("PLANE", font_small, WHITE, x + bar_width//2, 742, center=True)
                draw_text("(Hidden)", font_tiny, LIGHT_GRAY, x + bar_width//2, 770, center=True)

# ---------------------------------------------------------
# MAIN EVENT LOOP WITH VIEWPORT SCALING
# ---------------------------------------------------------
def main():
    clock = pygame.time.Clock()
    current_scene = MenuScene()
    update_viewport()

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                update_viewport()

        next_scene = current_scene.handle_events(events)
        if next_scene is not current_scene:
            current_scene = next_scene

        current_scene.update()
        current_scene.draw(virtual_surface)
        
        screen.fill((10, 10, 14))
        scaled_w = int(V_WIDTH * scale_factor)
        scaled_h = int(V_HEIGHT * scale_factor)
        scaled_surf = pygame.transform.smoothscale(virtual_surface, (scaled_w, scaled_h))
        screen.blit(scaled_surf, (offset_x, offset_y))
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()