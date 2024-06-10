from tkinter import *


def btn_clicked():
    print("Button Clicked")
    
    
def camera_clicked():
    import cam


def exam_history():
    import data_visualization


def attendance_clicked():
    #window.destroy()
    import attend

def head_pose_estimation():
    import head_pose_estimation

def eye_tracker_estimation():
    import eye_tracker

def audio_record():
    import audio_part


# window = Tk()

# window.geometry("1040x733")
# window.configure(bg = "#4d9982")
# canvas = Canvas(
#     window,
#     bg = "#4d9982",
#     height = 733,
#     width = 1040,
#     bd = 0,
#     highlightthickness = 0,
#     relief = "ridge")
# canvas.place(x = 0, y = 0)

# background_img = PhotoImage(file = f"img/b1.png")
# background = canvas.create_image(
#     520.0, 366.5,
#     image=background_img)

# img0 = PhotoImage(file = f"img/b2.png")
# b0 = Button(
#     image = img0,
#     borderwidth = 0,
#     highlightthickness = 0,
#     command = camera_clicked,
#     relief = "flat")

# b0.place(
#     x = 57, y = 195,
#     width = 24,
#     height = 24)

# img1 = PhotoImage(file = f"img/b3.png")
# b1 = Button(
#     image = img1,
#     borderwidth = 0,
#     highlightthickness = 0,
#     command = btn_clicked,
#     relief = "flat")

# b1.place(
#     x = 61, y = 307,
#     width = 24,
#     height = 24)

# img2 = PhotoImage(file = f"img/b4.png")
# b2 = Button(
#     image = img2,
#     borderwidth = 0,
#     highlightthickness = 0,
#     command = btn_clicked,
#     relief = "flat")

# b2.place(
#     x = 60, y = 145,
#     width = 24,
#     height = 24)

# img3 = PhotoImage(file = f"img/b5.png")
# b3 = Button(
#     image = img3,
#     borderwidth = 0,
#     highlightthickness = 0,
#     command = btn_clicked,
#     relief = "flat")

# b3.place(
#     x = 153, y = 495,
#     width = 66,
#     height = 36)

# img4 = PhotoImage(file = f"img/b6.png")
# b4 = Button(
#     image = img4,
#     borderwidth = 0,
#     highlightthickness = 0,
#     command = attendance_clicked,
#     relief = "flat")

# b4.place(
#     x = 61, y = 251,
#     width = 19,
#     height = 24)

# window.resizable(False, False)
# window.mainloop()

window = Tk()
window.geometry("1040x733")
window.title("DETECTION DE MOUVEMENT DE TRICHERIE DANS LES EXAMENS")
window.configure(bg = "white")

# Create label for Header
labelHead = Label(window, text="DETECTION DE MOUVEMENT DE TRICHERIE DANS LES EXAMENS", font=("Arial", 24), bg="orange") 
# labelHead.pack(padx=10, pady=10)

# Create a Label
label = Text(window, height=3, bg="white")
# label.grid(row=0, column=0)`1`
# label.pack(pady=10)




# Create a Button for camera
# camera_button = Button(window, text="Camera",font=("Arial", 24), command=camera_clicked, height=3, width=20 , bg="#21b2bb")
# # camera_button.grid(row=0, column=1)
# camera_button.grid(anchor=W,pady=10)

# # Create a Button for attendance
# historique_exam = Button(window,font=("Arial", 24), height=3, width=20 , bg="#21b2bb", text="Historique des examens", command=exam_history)
# historique_exam.grid(pady=0)

# Create a Button for camera
camera_button = Button(window, text="Camera",font=("Arial", 24), command=camera_clicked, height=3, width=20 , bg="#21b2bb")
camera_button.grid(row=1, column=0, pady=10, padx=30, sticky=E)

# Create a Button for attendance
historique_exam = Button(window,font=("Arial", 24), height=3, width=20 , bg="#21b2bb", text="Historique des examens", command=exam_history)
historique_exam.grid(row=1, column=1, pady=10, padx=150,sticky=E)


# Create a Button for HeadPosition
head_position_button = Button(window, text="Head Pose Estimation",font=("Arial", 24), command=head_pose_estimation, height=3, width=20 , bg="#21b2bb")
head_position_button.grid(row=2, column=0, pady=10, padx=30, sticky=E)


# Create a Button for attendance
eye_tracking = Button(window,font=("Arial", 24), height=3, width=20 , bg="#21b2bb", text="Eye Tracking ", command=eye_tracker_estimation)
eye_tracking.grid(row=2, column=1, pady=10, padx=150,sticky=E)

# Create a Button for Audio registration 
audio_registration = Button(window,font=("Arial", 24), height=3, width=20 , bg="#21b2bb", text="Audio Record ", command=audio_record)
audio_registration.grid(row=3, column=1, pady=10, padx=150,sticky=E+W)


window.mainloop()