#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linalg/matrix.hpp"
#include "linalg/misc.hpp"
#include "linalg/trans.hpp"
#include "shapes.hpp"
#include "typedefs.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

static constexpr int SCREEN_WIDTH  = 640;
static constexpr int SCREEN_HEIGHT = 400;

static float turn      = 0;
static bool  quit_game = false;

//////////////////////////////////////////////////////////////////////////////

auto grid_tmatt(float alpha, float x, float y) -> linalg::Matrixf<6, 6>
{

    return linalg::Matrixf<6, 6>{{{cosf(alpha), -sinf(alpha), 0, 0, 0, 0},
                                  {sinf(alpha), cosf(alpha), 0, 0, 0, 0},
                                  {x, y, 1, 0, 0, 0},
                                  {0, 0, 0, cosf(alpha), -sinf(alpha), 0},
                                  {0, 0, 0, sinf(alpha), cosf(alpha), 0},
                                  {0, 0, 0, x, y, 1}}};
}

//////////////////////////////////////////////////////////////////////////////

enum struct MouseState { default_state,
                         dragging };

struct MouseHandler {
    MouseState                    state;
    std::function<void(int, int)> on_mouse_click;
    std::function<void(int, int)> on_mouse_drag;
    std::function<void(int, int)> on_mouse_release;

    int mouse_prev_x;
    int mouse_prev_y;

    void handle_events()
    {
        int    mouse_x, mouse_y;
        uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);

        bool mouse_left_pressed = mouse_state & (SDL_BUTTON(SDL_BUTTON_LEFT));

        switch (state)
        {
        case MouseState::default_state: {
            if (mouse_left_pressed)
            {
                printf("mouse clicked\n");
                state = MouseState::dragging;
                on_mouse_click(mouse_x, mouse_y);

                mouse_prev_x = mouse_x;
                mouse_prev_y = mouse_y;
            }
        }
        case MouseState::dragging: {
            if (!mouse_left_pressed)
            {
                state = MouseState::default_state;

                on_mouse_release(mouse_x, mouse_y);

                mouse_prev_x = mouse_x;
                mouse_prev_y = mouse_y;
            }
            else
            {
                int dx = mouse_x - mouse_prev_x;
                int dy = mouse_y - mouse_prev_y;
                on_mouse_drag(dx, dy);

                mouse_prev_x = mouse_x;
                mouse_prev_y = mouse_y;
            }
        }
        }
    }
};

//////////////////////////////////////////////////////////////////////////////

void handle_input(MouseHandler& mouse_state)
{
    SDL_Event event;

    while (SDL_PollEvent(&event) != 0)
    {
        if (event.type == SDL_QUIT)
        {
        }
        else if (event.type == SDL_KEYUP)
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_UP:
                break;
            case SDLK_DOWN:
                break;
            case SDLK_LEFT:
                turn = 0;
                break;
            case SDLK_RIGHT:
                turn = 0;
                break;
            case SDLK_LSHIFT:
                break;
            default:
                break;
            }
        }
        else if (event.type == SDL_KEYDOWN)
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_UP:
                break;
            case SDLK_DOWN:
                break;
            case SDLK_LEFT:
                turn = -1;
                break;
            case SDLK_RIGHT:
                turn = 1;
                break;
            case SDLK_LSHIFT:
                break;
            case SDLK_ESCAPE:
                quit_game = true;
                break;
            default:
                break;
            }
        }
    } // End event loop

    mouse_state.handle_events();
}

//////////////////////////////////////////////////////////////////////////////

template <size_t Xs, size_t Ys, size_t NumSegs>
auto make_grid_points(float scale) -> std::vector<linalg::Matrixf<NumSegs + 2, 3>>
{
    constexpr int NumRows = Xs * Ys;

    linalg::Matrixf<NumRows, 2> points;

    int row = 0;

    for (auto i : irange<Xs>())
    {
        for (auto k : irange<Ys>())
        {
            points[row][0] = float(k);
            points[row][1] = float(i);
            ++row;
        }
    }

    points *= scale;

    std::vector<linalg::Matrixf<NumSegs + 2, 3>> circles;

    for (int i = 0; i < NumRows; ++i)
    {
        float x = points[i][0];
        float y = points[i][1];

        auto circle = make_circle_points<float, NumSegs>(10);
        auto transc = circle * rtransf(0, x, y);

        circles.push_back(transc);
    }

    return circles;
}

template <size_t Xs, size_t Ys>
auto make_grid_points(float scale) -> std::vector<linalg::Matrixf<8 + 2, 2>>
{
    return make_grid_points<Xs, Ys, 8>(scale);
}

//////////////////////////////////////////////////////////////////////////////

template <int NumLines>
auto make_grid_lines(float scale) -> linalg::Matrixf<(NumLines + 1) * 2, 6>
{
    static_assert((NumLines % 2) == 0);

    linalg::Matrixf<(NumLines + 1) * 2, 6> grid;

    const float length = NumLines;

    // Vertical lines
    for (auto i : irange<NumLines + 1>())
    {
        float fi = (float)i;
        span_deepcopy(grid[i], {0.f, fi, 1.f, length, fi, 1.f});
    }

    // Horizontal lines
    for (auto i : irange<NumLines + 1>())
    {
        float fi = (float)i;
        span_deepcopy(grid[i + NumLines + 1], {fi, 0.f, 1.f, fi, length, 1.f});
    }

    // Here we are Euler coordinates. Center the grid about (0,0).
    return grid * grid_tmatt(0, -NumLines / 2.f, -NumLines / 2.f);
}

//////////////////////////////////////////////////////////////////////////////
auto to_screen_y(float y) -> float
{
    return SCREEN_HEIGHT - y;
}

auto to_screen_y(int y) -> int
{
    return SCREEN_HEIGHT - y;
}

template <typename M>
void draw(SDL_Renderer* renderer, M& data)
{
    for (auto i : irange<0, M::rows() - 1>())
    {
        SDL_RenderDrawLineF(renderer,
                            data[i][0],
                            to_screen_y(data[i][1]),
                            data[i + 1][0],
                            to_screen_y(data[i + 1][1]));
    }
}

//////////////////////////////////////////////////////////////////////////////
static constexpr float scale = 40;

int main()
{
    srand(time(nullptr));

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("Failed to initialize.");
        return -1;
    }

    // Create window
    auto* window = SDL_CreateWindow("SDL Tutorial",
                                    SDL_WINDOWPOS_UNDEFINED,
                                    SDL_WINDOWPOS_UNDEFINED,
                                    SCREEN_WIDTH,
                                    SCREEN_HEIGHT,
                                    SDL_WINDOW_SHOWN);

    if (window == nullptr)
    {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create renderer
    auto* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr)
    {
        printf("Renderer could not be created! SDL_Error: %s\n",
               SDL_GetError());
        return -1;
    }

    // Init game objects
    SDL_Rect background_rect{0, 0, SCREEN_WIDTH, SCREEN_HEIGHT};

    float alpha = 0;
    float omega = 0;
    float theta = 0;


    const int NumLines   = 8;
    auto      grid_lines = make_grid_lines<NumLines>(scale);

    Circle<12> basis_x(0.25, 1, 0);
    Circle<12> basis_y(0.25, 0, 1);

    linalg::Matrixf<14, 2> selected_x;
    linalg::Matrixf<14, 2> selected_y;

    linalg::Vectorf<2> X{{0, 0}};
    linalg::Vectorf<2> Xdot{{0, 0}};

    bool clicked_x{};
    bool clicked_y{};

    float offset_x  = (SCREEN_WIDTH / 2) / scale;
    float offset_y  = (SCREEN_HEIGHT / 2) / scale;
    auto  rotated_x = basis_x.data * rtransf(0, basis_x.offset_x, basis_x.offset_y) * rtransf(basis_x.theta, offset_x, offset_y) * scale;
    auto  rotated_y = basis_y.data * rtransf(0, basis_y.offset_x, basis_y.offset_y) * rtransf(basis_y.theta, offset_x, offset_y) * scale;

    MouseHandler mouse_handler;
    mouse_handler.state          = MouseState::default_state;
    mouse_handler.on_mouse_click = [&rotated_x, &rotated_y, &clicked_x, &clicked_y](int mouse_x, int mouse_y) {
        float center_x;
        float center_y;
        float radius;

        center_x = rotated_x[0][0];
        center_y = to_screen_y(rotated_x[0][1]);
        radius   = std::sqrt(std::pow(center_x - rotated_x[1][0], 2)
                           + std::pow(center_y - to_screen_y(rotated_x[1][1]), 2));

        if ((std::abs(center_x - mouse_x) < radius) && (std::abs(center_y - mouse_y) < radius))
        {
            clicked_x = true;
            printf("Clicked x basis vector\n");
        }

        center_x = rotated_y[0][0];
        center_y = to_screen_y(rotated_y[0][1]);
        radius   = std::sqrt(std::pow(center_x - rotated_y[1][0], 2)
                           + std::pow(center_y - to_screen_y(rotated_y[1][1]), 2));

        if ((std::abs(center_x - mouse_x) < radius) && (std::abs(center_y - mouse_y) < radius))
        {
            clicked_y = true;
            printf("Clicked y basis vector\n");
        }
    };

    mouse_handler.on_mouse_drag = [&basis_x, &basis_y, &clicked_x, &clicked_y](int dx, int dy) {
        if (clicked_x)
        {
            basis_x.offset_x += dx / scale;
            basis_x.offset_y -= dy / scale;
        }
        else if (clicked_y)
        {
            basis_y.offset_x += dx / scale;
            basis_y.offset_y -= dy / scale;
        }
    };

    mouse_handler.on_mouse_release = [&clicked_x, &clicked_y](int, int) {
        clicked_x = false;
        clicked_y = false;
    };

    while (!quit_game)
    {
        handle_input(mouse_handler);

        // Clear the screen
        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderFillRect(renderer, &background_rect);

        float u  = turn * 5;
        float dt = 1 / 30.f;
        float m  = 1.f;
        float k  = 0.2f;
        float v  = X[1];

        // 0.2 tanh(10x)+x^3/10;
        linalg::Matrixf<2, 3> A{{{0, 1, 0},
                                 {0, (-powf(v, 2) / 10.f * m), -tanhf(10 * v)}}};
        linalg::Matrixf<2, 2> I{{{1, 0},
                                 {0, 1}}};
        linalg::Matrixf<2, 1> B{{{0, 1}}};

        linalg::Matrixf<3, 1> Xn{{{X[0], X[1], 1}}};

        Xdot = (X + (dt * A * Xn)) + ((dt * B) * u);

        linalg::Matrixf<1, 2> C{{{1, 0}}};
        linalg::Matrixf<1, 1> y = C * X;

        X     = Xdot;
        theta = y[0];

        basis_x.theta = theta;
        basis_y.theta = theta;

        rotated_x = basis_x.data * rtransf(0, basis_x.offset_x, basis_x.offset_y) * rtransf(basis_x.theta, offset_x, offset_y) * scale;
        rotated_y = basis_y.data * rtransf(0, basis_y.offset_x, basis_y.offset_y) * rtransf(basis_y.theta, offset_x, offset_y) * scale;

        float x1 = basis_x.offset_x;
        float x2 = basis_y.offset_x;
        float y1 = basis_x.offset_y;
        float y2 = basis_y.offset_y;

        // printf("%f  %f  %f  %f\n", x1, x2, y1, y2);

        linalg::Matrixf<6, 6> basis{{{x1, x2, 0, 0, 0, 0},
                                     {y1, y2, 0, 0, 0, 0},
                                     {0, 0, 1, 0, 0, 0},
                                     {0, 0, 0, x1, x2, 0},
                                     {0, 0, 0, y1, y2, 0},
                                     {0, 0, 0, 0, 0, 1}}};

        // Draw the grid.
        {
            float offset_x = SCREEN_WIDTH / 2 / scale;
            float offset_y = SCREEN_HEIGHT / 2 / scale;

            SDL_SetRenderDrawColor(renderer, 0x50, 0x50, 0x50, 0x00);
            auto background_grid = grid_lines * grid_tmatt(0, offset_x, offset_y) * scale;
            for (auto r : iter(background_grid))
            {
                SDL_RenderDrawLineF(renderer,
                                    r[0],
                                    to_screen_y(r[1]),
                                    r[3],
                                    to_screen_y(r[4]));
            }

            SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0x00);
            auto rotated_grid = (grid_lines * basis) * grid_tmatt(theta, offset_x, offset_y) * scale;

            // printf("%f\n", (basis_x.offset_x - (SCREEN_WIDTH / 2)) / scale);

            for (auto r : iter(rotated_grid))
            {
                SDL_RenderDrawLineF(renderer,
                                    r[0],
                                    to_screen_y(r[1]),
                                    r[3],
                                    to_screen_y(r[4]));
            }
        }

        // Draw the basis point.
        {
            SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);

            draw(renderer, rotated_x);
            draw(renderer, rotated_y);
        }

        // Update the screen with rendering actions
        SDL_RenderPresent(renderer);

        SDL_Delay(33);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
