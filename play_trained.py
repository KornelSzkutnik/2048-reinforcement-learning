# Play 2048 using a trained TD agent loaded from `agent_2048.pkl`.

import pygame

from board import Board
from game_logic_2048 import raw_board_to_log2
from td_agent import TDAgent

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SEPARATOR = 5
BLOCK_SIZE = (SCREEN_WIDTH + SEPARATOR * 3) // 4

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

AGENT_FILE = "agent_2048.pkl"

# Map directions: TD agent (0=left, 1=up, 2=right, 3=down) -> Board (0=left, 1=right, 2=up, 3=down)
AGENT_TO_BOARD_ACTION = {0: 0, 1: 2, 2: 1, 3: 3}

agent = TDAgent.load_agent(AGENT_FILE)
print(f"Agent: {agent.name}, n={agent.n}, top_score={agent.top_score}")
print("Press SPACE for the AI to make a move")
print("Press 'A' to toggle AUTO mode")
print("Press 'R' to reset the game")

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2048 - Trained TD Agent")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)
score_font = pygame.font.Font(None, 40)
running = True

board = Board()
auto_mode = False
moves_count = 0
action_names = ["LEFT", "RIGHT", "UP", "DOWN"]


def get_ai_action():
    """
    Convert the current board to log2 representation and ask the agent
    for the best direction (0=left, 1=up, 2=right, 3=down). Then map it to
    `Board`'s action convention (0=left, 1=right, 2=up, 3=down).
    """
    state_log2 = raw_board_to_log2(board.board)
    direction, _ = agent.get_best_action(state_log2, board.score)
    return AGENT_TO_BOARD_ACTION[direction]


def execute_action(action):
    """
    Execute a move on the raw `Board`.

    action: 0=left, 1=right, 2=up, 3=down.
    Returns (moved, reward).
    """
    if action == 0:
        return board.move_left()
    if action == 1:
        return board.move_right()
    if action == 2:
        return board.move_up()
    return board.move_down()


def draw_tile(screen, value, x, y, font):
    pygame.draw.rect(
        screen, COLORS.get(value, COLORS[0]), [x, y, BLOCK_SIZE - SEPARATOR, BLOCK_SIZE - SEPARATOR]
    )
    if value != 0:
        text = font.render(str(value), True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2))
        screen.blit(text, text_rect)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not auto_mode:
                action = get_ai_action()
                moved, reward = execute_action(action)
                if moved:
                    board.generate()
                    moves_count += 1
                    print(
                        f"Move {moves_count}: Score={board.score}, Reward={reward}, "
                        f"Max tile={board.get_max_tile()} ({action_names[action]})"
                    )
                if board.is_game_over():
                    print("\n=== GAME OVER ===")
                    print(
                        f"Final Score: {board.score}, "
                        f"Max Tile: {board.get_max_tile()}, Moves: {moves_count}"
                    )
                    print("Press R to play again")
                elif board.has_won():
                    print(f"\n=== WIN! 2048 achieved! Score: {board.score} ===")

            if event.key == pygame.K_a:
                auto_mode = not auto_mode
                print(f"AUTO mode: {'ON' if auto_mode else 'OFF'}")

            if event.key == pygame.K_r:
                board.reset()
                moves_count = 0
                print("\n=== NEW GAME ===")

    if auto_mode and not board.is_game_over() and not board.has_won():
        action = get_ai_action()
        moved, reward = execute_action(action)
        if moved:
            board.generate()
            moves_count += 1
            if moves_count % 10 == 0:
                print(f"Move {moves_count}: Score={board.score}, Max tile={board.get_max_tile()}")
        if board.is_game_over():
            print(
                f"\n=== GAME OVER === "
                f"Score: {board.score}, Max: {board.get_max_tile()}, Moves: {moves_count}"
            )
            auto_mode = False
        elif board.has_won():
            print(f"\n=== WIN! Score: {board.score} ===")
            auto_mode = False
        pygame.time.delay(100)

    screen.fill("black")
    for i in range(4):
        for j in range(4):
            pygame.draw.rect(
                screen,
                COLORS[0],
                [i * BLOCK_SIZE, j * BLOCK_SIZE, BLOCK_SIZE - SEPARATOR, BLOCK_SIZE - SEPARATOR],
            )
    for i in range(4):
        for j in range(4):
            value = board.board[j][i]
            if value != 0:
                draw_tile(screen, value, i * BLOCK_SIZE, j * BLOCK_SIZE, font)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
