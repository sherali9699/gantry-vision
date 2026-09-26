import cv2
import numpy as np

squares_x = 6
squares_y = 4
square_length = 0.03  # 3cm squares → board is 18cm wide, fits A4 with margin
marker_length = 0.022  # ~75% of square_length

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    dictionary
)

img = board.generateImage((2480, 1754), marginSize=20)  # A4 at 300dpi
cv2.imwrite("charuco_board.png", img)
print("Saved charuco_board.png")