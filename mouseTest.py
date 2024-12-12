import pygame
import pyautogui

# Initialize pygame
pygame.init()

# Set up joystick
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    exit()

joystick = pygame.joystick.Joystick(0)  # Assuming the first joystick
joystick.init()

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Mouse sensitivity
sensitivity = 10

try:
    while True:
        # Event loop to capture joystick inputs
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # Read joystick axes (typically axes 0 and 1 for X and Y)
                x_axis = joystick.get_axis(0)  # Left/Right
                y_axis = joystick.get_axis(1)  # Up/Down

                # Map joystick movement to mouse movement
                current_mouse_x, current_mouse_y = pyautogui.position()
                new_mouse_x = current_mouse_x + int(x_axis * sensitivity)
                new_mouse_y = current_mouse_y + int(y_axis * sensitivity)

                # Ensure mouse stays within screen bounds
                new_mouse_x = max(0, min(screen_width - 1, new_mouse_x))
                new_mouse_y = max(0, min(screen_height - 1, new_mouse_y))

                # Move mouse
                pyautogui.moveTo(new_mouse_x, new_mouse_y)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    pygame.quit()
