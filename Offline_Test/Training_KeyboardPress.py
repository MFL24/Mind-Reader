import pyautogui
import time

T = 1.5
N = 10

for i in range(N):
    pyautogui.press('a')
    time.sleep(T)



# import pyautogui
# import time

# def press_a_key_every_2_seconds():
#     try:
#         while True:
#             pyautogui.press('a')
#             time.sleep(1a)
#     except KeyboardInterrupt:
#         print("Program terminated by user.")

# if __name__ == "__main__":
#     press_a_key_every_2_seconds()




