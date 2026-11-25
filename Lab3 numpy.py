import numpy as np

print("--- Task 1 ---")

# 1a
temps = np.array([31, 28, 35, 30, 33, 29, 36])
print("1a temps:", temps)
print("dtype:", temps.dtype)
print("shape:", temps.shape)
print("size:", temps.size)

# 1b
months = np.arange(1, 13).reshape(3, 4)
print("\n1b months:")
print(months)
print("shape:", months.shape)

# 1c
five_zeros = np.zeros(5)
ones_2x3 = np.ones((2, 3))
print("\n1c Five zeros:", five_zeros)
print("2x3 ones:")
print(ones_2x3)

# 1d
rng = np.random.default_rng(42)
dice = rng.integers(1, 7, size=10)
print("\n1d Dice rolls (seed 42):", dice)

print("\n--- Task 2 ---")

# 2a - First and last
print("2a First:", temps[0], "Last:", temps[-1])

# 2b - Weekend (last two days)
print("2b Weekend:", temps[-2:])

# 2c - Reverse order
print("2c Reverse:", temps[::-1])

# 2d - Hot days (above 32), count and average - No loops!
hot_days = temps[temps > 32]
print("2d Hot days:", hot_days)
print("How many:", hot_days.size)
print("Average:", hot_days.mean())

# 2e - From 29 to 33 (both included)
print("2e 29 to 33:", temps[(temps >= 29) & (temps <= 33)])

# 2f - Label Hot (33 or more) or OK
labels = np.where(temps >= 33, "Hot", "OK")
print("2f Labels:", labels)

print("\n--- Task 3 ---")
names = np.array(["Ali", "Usman", "Hina", "Sara", "Ahmed"])

marks = np.array([
    [70, 80, 75, 85],
    [60, 65, 70, 72],
    [90, 85, 88, 92],
    [55, 60, 65, 70],
    [80, 75, 78, 82]
])

# 3a - Shape of marks
print("3a Shape:", marks.shape)

# 3b 
print("3b Usman marks:", marks[1])
print("3b Subject 3 marks:", marks[:, 2])

# 3c 
print("3c Subject average:", marks.mean(axis=0))
print("3c Student average:", marks.mean(axis=1))

# 3d
student_avg = marks.mean(axis=1)
top_student = np.argmax(student_avg)
print("3d Top student:", names[top_student])

# 3e 
weights = np.array([0.3, 0.3, 0.2, 0.2])
weighted_total = np.round((marks * weights).sum(axis=1), 1)
print("3e Weighted total:", weighted_total)

# 3f 
new_marks = np.clip(marks + 5, 0, 100)
print("3f New marks:")
print(new_marks)

# 3g 
failed = (marks < 50).sum(axis=0)
print("3g Failed in each subject:", failed)

import numpy as np

print("\n--- Task 4 ---")

# Image as a 5 x 5 array (plus sign)
img = np.array([
    [10, 10, 250, 10, 10],
    [10, 10, 250, 10, 10],
    [250, 250, 250, 250, 250],
    [10, 10, 250, 10, 10],
    [10, 10, 250, 10, 10]
])

# 4a - Shape, darkest and brightest value
print("4a Shape:", img.shape)
print("4a Darkest:", img.min())
print("4a Brightest:", img.max())

# 4b - Inverted image
inverted = 255 - img
print("4b Inverted:")
print(inverted)

# 4c - Black and white image
bw = (img > 128).astype(int)
print("4c Black-and-white:")
print(bw)

# 4d - Row 2 normalized to 0 to 1
row2_norm = np.round(img[2] / 255.0, 2)
print("4d Row 2 normalized:", row2_norm)

# 4e - 3x3 Centre crop
centre = img[1:4, 1:4]
print("4e 3x3 Centre:")
print(centre)