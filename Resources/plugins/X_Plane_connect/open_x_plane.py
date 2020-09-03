# Keyboard
from pynput.mouse import Button, Controller
from time import sleep

sleep(10)
mouse = Controller()

mouse.position = (1, 1)
mouse.press(Button.left)
sleep(1)

mouse.position = (1079, 1979)
mouse.press(Button.left)
sleep(1)

mouse.position = (300, 400)
mouse.press(Button.left)
sleep(0.1)
mouse.release(Button.left)
sleep(0.1)


print("Mouse Success")