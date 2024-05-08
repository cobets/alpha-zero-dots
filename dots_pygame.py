from dotsenv import DotsEnv, BLACK, RED, ENABLED, DISABLED
import pygame

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_DISABLED_BLACK = (128, 128, 128)
COLOR_DISABLED_RED = (255, 200, 200)

COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_LINE = (150, 150, 220)


WIDTH = 480
HEIGHT = 480
GRID_SPACE = 67
GRID_MARGIN = 5
LINE_WIDTH = 1


def draw_grid(surface):
    # draw vertical lines
    x = GRID_MARGIN
    while x < WIDTH:
        pygame.draw.line(surface, COLOR_LINE, (x, GRID_MARGIN), (x, HEIGHT - GRID_MARGIN - LINE_WIDTH), LINE_WIDTH)
        x += GRID_SPACE
    # draw horizontal lines
    y = GRID_MARGIN
    while y < HEIGHT:
        pygame.draw.line(surface, COLOR_LINE, (GRID_MARGIN, y), (WIDTH - GRID_MARGIN - LINE_WIDTH, y), LINE_WIDTH)
        y += GRID_SPACE


def get_action(env, pos):
    x, y = pos
    x = int((x + GRID_SPACE / 2) // GRID_SPACE)
    y = int((y + GRID_SPACE / 2) // GRID_SPACE)
    return x * env.width + y


def draw_line(surface, color, start_point, end_point):
    x_start, y_start = start_point
    x_end, y_end = end_point
    pygame.draw.line(
        surface=surface,
        color=color,
        start_pos=(x_start * GRID_SPACE + GRID_MARGIN, y_start * GRID_SPACE + GRID_MARGIN),
        end_pos=(x_end * GRID_SPACE + GRID_MARGIN, y_end * GRID_SPACE + GRID_MARGIN),
        width=3
    )


def draw_border(surface, paths, color):
    for path in paths:
        if path:
            l_first_point = None
            l_prev_point = None

            for point in path:
                if l_first_point is None:
                    l_first_point = point
                else:
                    draw_line(surface, color, l_prev_point, point)
                l_prev_point = point

            draw_line(surface, color, l_prev_point, l_first_point)


def draw(env, surface, font):
    surface.fill(COLOR_WHITE)
    draw_grid(surface)
    for x in range(env.width):
        for y in range(env.height):
            color = None
            v = env.board[x, y]
            if v & (BLACK | ENABLED) == (BLACK | ENABLED):
                color = COLOR_BLACK
            elif v & (RED | ENABLED) == (RED | ENABLED):
                color = COLOR_RED
            elif v & (BLACK | DISABLED) == (BLACK | DISABLED):
                color = COLOR_DISABLED_BLACK
            elif v & (RED | DISABLED) == (RED | DISABLED):
                color = COLOR_DISABLED_RED

            if color is not None:
                pygame.draw.circle(
                    surface,
                    color,
                    (x * GRID_SPACE + GRID_MARGIN, y * GRID_SPACE + GRID_MARGIN),
                    6
                )
    img = font.render(str(env.last_catch_area_size), True, COLOR_RED)
    surface.blit(img, (20, 20))

    # draw borders
    draw_border(surface, env.caught_paths_black, COLOR_BLACK)
    draw_border(surface, env.caught_paths_red, COLOR_RED)


def play(env, pos):
    action = get_action(env, pos)
    l_result = action in (env.legal_actions() if callable(env.legal_actions) else env.legal_action_values())

    if l_result:
        env.play(action)

    return l_result


def main(env, ai_play=None):
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption('Dots Game')

    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("", 16)

    draw(env, surface, font)

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                case pygame.MOUSEBUTTONDOWN:
                    if play(env, pygame.mouse.get_pos()):
                        draw(env, surface, font)

                        if ai_play is not None:
                            pygame.display.update()
                            ai_play()
                            draw(env, surface, font)

            pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    main(DotsEnv(8, 8, True))
