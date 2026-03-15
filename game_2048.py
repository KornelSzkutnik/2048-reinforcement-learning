# Main script for manual 2048 gameplay - a clean version of the game for a human player.

import pygame

from board import Board

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SEPARATOR = 5
BLOCK_SIZE = (SCREEN_WIDTH + SEPARATOR * 3) // 4
BOARD = Board()

COLORS = {
    0: (224, 224, 224, 255),
    2: (224, 229, 204, 255),
    4: (255, 204, 153, 255),
    8: (255, 178, 102, 255),
    16: (255, 153, 51, 255),
    32: (255, 128, 153, 255),
    64: (204, 102, 153, 255),
    128: (255, 255, 153, 255),
    256: (255, 255, 0, 255),
    512: (153, 255, 153, 255),
    1024: (51, 255, 51, 255),
    2048: (0, 153, 153, 255),
}

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)
score_font = pygame.font.Font(None, 40)
running = True

# Animation state
animating = False
animation_progress = 0
old_board = None
animation_direction = None


def draw_tile(value, x, y):
    """Draw a single tile."""
    pygame.draw.rect(screen, COLORS[value], [x, y, BLOCK_SIZE - SEPARATOR, BLOCK_SIZE - SEPARATOR])
    if value != 0:
        text = font.render(str(value), True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2))
        screen.blit(text, text_rect)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and not animating:
            moved = False
            reward = 0
            old_board = BOARD.board.copy()

            if event.key == pygame.K_LEFT:
                moved, reward = BOARD.move_left()
                animation_direction = "left"
            if event.key == pygame.K_RIGHT:
                moved, reward = BOARD.move_right()
                animation_direction = "right"
            if event.key == pygame.K_UP:
                moved, reward = BOARD.move_up()
                animation_direction = "up"
            if event.key == pygame.K_DOWN:
                moved, reward = BOARD.move_down()
                animation_direction = "down"

            if moved:
                animating = True
                animation_progress = 0
                print(f"Score: {BOARD.score}, Reward: {reward}")

            if BOARD.has_won():
                print(f"WIN! You’ve reached 2048! Final score: {BOARD.score}")
                running = False
            elif BOARD.is_game_over():
                print(f"GAME OVER! No moves available. Final score: {BOARD.score}")
                running = False

    if animating:
        animation_progress += 0.25
        if animation_progress >= 1.0:
            animating = False
            animation_progress = 1.0
            BOARD.generate()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    for i in range(4):
        for j in range(4):
            pygame.draw.rect(
                screen,
                COLORS[0],
                [i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE - SEPARATOR, BLOCK_SIZE - SEPARATOR],
            )

    # RENDER YOUR GAME HERE
    if animating and old_board is not None:
        for j in range(4):
            for i in range(4):
                value = old_board[j][i]
                if value != 0:
                    offset_x = 0
                    offset_y = 0

                    if animation_direction == "left":
                        target_i = i
                        for check_i in range(i, -1, -1):
                            if BOARD.board[j][check_i] != 0:
                                target_i = check_i
                                break
                        offset_x = (target_i - i) * BLOCK_SIZE * animation_progress

                    elif animation_direction == "right":
                        target_i = i
                        for check_i in range(i, 4):
                            if BOARD.board[j][check_i] != 0:
                                target_i = check_i
                                break
                        offset_x = (target_i - i) * BLOCK_SIZE * animation_progress

                    elif animation_direction == "up":
                        target_j = j
                        for check_j in range(j, -1, -1):
                            if BOARD.board[check_j][i] != 0:
                                target_j = check_j
                                break
                        offset_y = (target_j - j) * BLOCK_SIZE * animation_progress

                    elif animation_direction == "down":
                        target_j = j
                        for check_j in range(j, 4):
                            if BOARD.board[check_j][i] != 0:
                                target_j = check_j
                                break
                        offset_y = (target_j - j) * BLOCK_SIZE * animation_progress

                    draw_tile(value, i * BLOCK_SIZE + offset_x, j * BLOCK_SIZE + offset_y)
    else:
        for i in range(4):
            for j in range(4):
                value = BOARD.board[j][i]
                if value != 0:
                    draw_tile(value, i * BLOCK_SIZE, j * BLOCK_SIZE)

    score_text = score_font.render(f"Score: {BOARD.score}", True, (255, 255, 255))
    screen.blit(score_text, (10, SCREEN_HEIGHT - 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
