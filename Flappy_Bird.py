import pygame
import mysql.connector
import random
import os
from dotenv import load_dotenv

# ---------- PATHS ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_FOLDER = "assets" if os.path.exists(os.path.join(BASE_DIR, "assets")) else "assests"
SOUND_DIR = os.path.join(BASE_DIR, ASSET_FOLDER, "sounds")
IMAGE_DIR = os.path.join(BASE_DIR, ASSET_FOLDER, "images")

pygame.init()
pygame.mixer.init()

# ---------- ASSET PATHS ----------
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")
#SOUND_DIR = os.path.join(BASE_DIR, "assets", "sounds")

# ---------- DATABASE ----------

load_dotenv(os.path.join(BASE_DIR, ".env"))

db = mysql.connector.connect(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    database="flappy"
)

cursor = db.cursor()


cursor.execute("""
    CREATE TABLE IF NOT EXISTS scores (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        score INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
db.commit()


def save_score(username, score):
    """Insert a finished run into the database."""
    cursor.execute(
        "INSERT INTO scores (username, score) VALUES (%s, %s)",
        (username, score)
    )
    db.commit()


def get_high_score():
    """Return (username, score) of the single highest score ever recorded, or None."""
    cursor.execute(
        "SELECT username, score FROM scores ORDER BY score DESC LIMIT 1"
    )
    row = cursor.fetchone()
    return row if row else None


WIDTH, HEIGHT = 720, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

# ---------- LOAD SOUNDS ----------

flap_sound = pygame.mixer.Sound(os.path.join(SOUND_DIR, "flap.wav"))
point_sound = pygame.mixer.Sound(os.path.join(SOUND_DIR, "point.wav"))
hit_sound = pygame.mixer.Sound(os.path.join(SOUND_DIR, "hit.wav"))
die_sound = pygame.mixer.Sound(os.path.join(SOUND_DIR, "die.wav"))
swoosh_sound = pygame.mixer.Sound(os.path.join(SOUND_DIR, "swoosh.wav"))

flap_sound.set_volume(0.5)
point_sound.set_volume(0.5)
hit_sound.set_volume(0.6)
die_sound.set_volume(0.7)
swoosh_sound.set_volume(0.4)

# ---------- BIRD SPRITES ----------
bird_up = pygame.image.load(os.path.join(IMAGE_DIR, "flappy_up.png")).convert_alpha()
bird_down = pygame.image.load(os.path.join(IMAGE_DIR, "flappy_down.png")).convert_alpha()
bird_up = pygame.transform.scale(bird_up, (40, 30))
bird_down = pygame.transform.scale(bird_down, (40, 30))

bird_x = 80
bird_y = HEIGHT // 2    
velocity = 0
gravity = 0.4

wing_timer = 0
wing_state = "up"

# ---------- COIN SPRITE ----------
coin_img = pygame.image.load(os.path.join(IMAGE_DIR, "coin.png")).convert_alpha()
coin_img = pygame.transform.scale(coin_img, (40, 40))
coin_x = -100
coin_y = random.randint(200, 300)
coin_visible = False
coin_spawned = False

# ---------- GAME STATE ----------
# "name_entry" -> "ready" -> "playing" -> "over"
game_state = "name_entry"
game_over = False
score_saved = False
death_time = 0

# ---------- NAME INPUT BOX ----------
player_name = ""
name_active = True
input_box = pygame.Rect(WIDTH // 2 - 160, HEIGHT // 2 - 25, 320, 50)
cursor_visible = True
cursor_timer = 0
MAX_NAME_LEN = 16

COL_BOX_BG = (255, 255, 255)
COL_BOX_BORDER_ACTIVE = (255, 200, 0)
COL_BOX_BORDER = (60, 60, 60)
COL_TEXT = (30, 30, 30)
COL_PLACEHOLDER = (150, 150, 150)
COL_SHADOW = (0, 0, 0, 60)

# ---------- PIPES ----------
pipe_width = 60
pipe_x = WIDTH
pipe_gap = 150
top_height = random.randint(100, 350)

# ---------- SCORE ----------
score = 0
passed_pipe = False
font = pygame.font.SysFont("Arial", 40)
small_font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 60)
name_font = pygame.font.SysFont("Arial", 30)
title_font = pygame.font.SysFont("Arial", 36, bold=True)

# ---------- HIGH SCORE  ----------
high_score_row = get_high_score()  

# ---------- CLOUDS ----------
WHITE = (255, 255, 255)
clouds = []
for _ in range(4):
    clouds.append([
        random.randint(0, WIDTH),
        random.randint(0, HEIGHT // 2),
        random.uniform(0.5, 1.5)
    ])


def draw_cloud(x, y):
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), 20)
    pygame.draw.circle(screen, WHITE, (int(x + 25), int(y - 10)), 25)
    pygame.draw.circle(screen, WHITE, (int(x + 55), int(y)), 20)
    pygame.draw.circle(screen, WHITE, (int(x + 25), int(y + 10)), 20)


def draw_high_score_banner():
    """Small banner shown in the corner during name entry / ready / playing."""
    if high_score_row:
        hs_name, hs_score = high_score_row
        text = f"High Score: {hs_score} ({hs_name})"
    else:
        text = "High Score: --"
    hs_surface = small_font.render(text, True, (0, 0, 0))
    screen.blit(hs_surface, (WIDTH - hs_surface.get_width() - 10, 10))


def draw_name_entry_screen():
   
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 40))
    screen.blit(overlay, (0, 0))

    title_surface = title_font.render("Enter Your Name", True, (20, 20, 20))
    screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2,
                                 input_box.y - 60))

  
    card_rect = input_box.inflate(40, 40)
    card_surface = pygame.Surface((card_rect.width, card_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(card_surface, (255, 255, 255, 230), card_surface.get_rect(), border_radius=16)
    screen.blit(card_surface, card_rect.topleft)

    # Input box itself
    border_color = COL_BOX_BORDER_ACTIVE if name_active else COL_BOX_BORDER
    pygame.draw.rect(screen, COL_BOX_BG, input_box, border_radius=10)
    pygame.draw.rect(screen, border_color, input_box, width=3, border_radius=10)

    if player_name:
        text_surface = name_font.render(player_name, True, COL_TEXT)
    else:
        text_surface = name_font.render("Type your name...", True, COL_PLACEHOLDER)
    screen.blit(text_surface, (input_box.x + 15, input_box.y + (input_box.height - text_surface.get_height()) // 2))

    # Blinking cursor
    if name_active and cursor_visible and player_name:
        cursor_x = input_box.x + 15 + text_surface.get_width() + 2
        cursor_y1 = input_box.y + 10
        cursor_y2 = input_box.y + input_box.height - 10
        pygame.draw.line(screen, COL_TEXT, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)

    hint_surface = small_font.render("Press ENTER to continue", True, (70, 70, 70))
    screen.blit(hint_surface, (WIDTH // 2 - hint_surface.get_width() // 2, input_box.bottom + 20))


# ---------- GAME LOOP ----------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if game_state == "name_entry" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if player_name.strip():
                    player_name = player_name.strip()
                    game_state = "ready"
            elif event.key == pygame.K_BACKSPACE:
                player_name = player_name[:-1]
            else:
                if len(player_name) < MAX_NAME_LEN and event.unicode.isprintable():
                    player_name += event.unicode

        elif game_state in ("ready", "playing") and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if game_state == "ready":
                game_state = "playing"
                velocity = 0
                swoosh_sound.play()
            else:
                velocity = -7
                flap_sound.play()

    if game_state == "playing":
        velocity += gravity
        bird_y += velocity
        pipe_x -= 3
        if coin_spawned:
            coin_x -= 3

    # Wing animation
    wing_timer += 1
    if wing_timer > 8:
        wing_state = "down" if wing_state == "up" else "up"
        wing_timer = 0

    # Cursor blink timer
    cursor_timer += 1
    if cursor_timer > 30:
        cursor_visible = not cursor_visible
        cursor_timer = 0

    if game_state == "playing":
        # Pipe scoring
        if (pipe_x + pipe_width) < bird_x and not passed_pipe:
            score += 1
            passed_pipe = True
            point_sound.play()

        # Reset pipe & coin
        if pipe_x < -pipe_width:
            pipe_x = WIDTH
            top_height = random.randint(100, 350)
            passed_pipe = False

            if random.random() < 0.8:
                coin_x = WIDTH + 150
                coin_y = random.randint(100, HEIGHT - 100)
                coin_visible = True
                coin_spawned = True
            else:
                coin_visible = False
                coin_spawned = False
                coin_x = -100

        if coin_x < -50:
            coin_visible = False
            coin_spawned = False

        # Collision rectangles
        bird_rect = pygame.Rect(bird_x, bird_y, 40, 30)
        top_pipe_rect = pygame.Rect(pipe_x, 0, pipe_width, top_height)
        bottom_pipe_rect = pygame.Rect(pipe_x, top_height + pipe_gap, pipe_width, HEIGHT)

        # Coin collision (+5 points)
        if coin_visible and coin_spawned:
            coin_rect = pygame.Rect(coin_x, coin_y, 40, 40)
            if bird_rect.colliderect(coin_rect):
                score += 5
                coin_visible = False
                coin_spawned = False
                point_sound.play()

        # Death collision
        if (bird_rect.colliderect(top_pipe_rect)
                or bird_rect.colliderect(bottom_pipe_rect)
                or bird_y <= 0
                or bird_y + 30 >= HEIGHT):

            hit_sound.play()
            pygame.time.delay(150)
            die_sound.play()
            game_state = "over"
            death_time = pygame.time.get_ticks()
    else:
        
        bird_rect = pygame.Rect(bird_x, bird_y, 40, 30)

    if game_state == "over" and not score_saved:
        save_score(player_name, score)
        high_score_row = get_high_score()
        score_saved = True

    # ---------- DRAW ----------
    screen.fill((28, 226, 240))

    for cloud in clouds:
        draw_cloud(cloud[0], cloud[1])
        cloud[0] += cloud[2]
        if cloud[0] > WIDTH + 60:
            cloud[0] = random.randint(-200, -60)
            cloud[1] = random.randint(0, HEIGHT // 2)

    if game_state != "name_entry":
        # Pipes
        pygame.draw.rect(screen, (82, 220, 82), (pipe_x, 0, pipe_width, top_height))
        pygame.draw.rect(screen, (82, 220, 82), (pipe_x, top_height + pipe_gap, pipe_width, HEIGHT))

        # Coin
        if coin_visible and coin_spawned:
            screen.blit(coin_img, (coin_x, coin_y))

        # Bird
        if wing_state == "up":
            screen.blit(bird_up, (bird_x, bird_y))
        else:
            screen.blit(bird_down, (bird_x, bird_y))

        # Current score
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        draw_high_score_banner()

    if game_state == "ready":
        start_text = font.render("Press SPACE to Start", True, (0, 0, 0))
        screen.blit(start_text, (WIDTH // 2 - 170, HEIGHT // 2 - 20))
        name_tag = small_font.render(f"Player: {player_name}", True, (0, 0, 0))
        screen.blit(name_tag, (WIDTH // 2 - name_tag.get_width() // 2, HEIGHT // 2 + 30))

    if game_state == "name_entry":
        draw_name_entry_screen()
        draw_high_score_banner()

    if game_state == "over":
        lose_text = big_font.render("YOU LOSE", True, (255, 0, 0))
        final_score = font.render(f"Final Score: {score}", True, (0, 0, 0))
        screen.blit(lose_text, (WIDTH // 2 - 130, HEIGHT // 2 - 80))
        screen.blit(final_score, (WIDTH // 2 - 110, HEIGHT // 2 - 20))

        if high_score_row:
            hs_name, hs_score = high_score_row
            hs_text = font.render(f"High Score: {hs_score} ({hs_name})", True, (0, 100, 0))
            screen.blit(hs_text, (WIDTH // 2 - hs_text.get_width() // 2, HEIGHT // 2 + 30))

        if pygame.time.get_ticks() - death_time > 5000:
            running = False

    pygame.display.update()
    clock.tick(60)

cursor.close()
db.close()
pygame.quit()