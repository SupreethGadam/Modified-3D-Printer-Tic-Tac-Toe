import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import queue
import threading



class EnhancedConvNet(nn.Module):
    def __init__(self): 
        super(EnhancedConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*3*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.batchnorm1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.batchnorm2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.batchnorm3(self.conv3(x)), 2))
        x = x.view(-1, 128*3*3)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


##### Pytorch model trained on MNist

model = EnhancedConvNet()


# Load the trained weights into the model
model_path = "C:/Users/supre/OneDrive/Desktop/Codes/mnist_model2.pth" # to update to github location
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Transformations to suit  (similar to what was used during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

## Capturing image from webcam

def capture_image_from_webcam(cap):
    
    print("Press 'Space' to capture the image")

    dual_camera= 0
    #Video properties
    half_width = int(cap.get(3))//2  
    half_height = int(cap.get(4))//2   
    
    while True:
        ret, frame = cap.read()

        if dual_camera ==1:         #for stereo camera
            frame = frame[:, :half_width]
        
        if not ret or frame is None:
            print("Failed to capture image")               
            continue  
        # Inverting frame about x-axis
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1500) or 0xFF == ord(' '):
            image = frame.copy()
            break

    if dual_camera == 1: # if using stereo camera
        image= image[:,:half_width]
        image = image[:half_height, :] 
        
    #cap.release()
    cv2.destroyAllWindows()
    return image

###############################################
## Image Pre-Processing
###############################################

def preprocess_image(captured_image):
    
    img = captured_image
    img = cv2.GaussianBlur(img,(1,1),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    cv2.imshow('Capture', res2)
    cv2.waitKey(300)  # Wait for a key press
    cv2.destroyAllWindows()

    return res, res2, mask, img

##
## Finding and creating largest mask image
##

def create_square_mask_image(processed_image, mask):
    thresh = cv2.adaptiveThreshold(processed_image,255,0,1,19,2)
    contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    img_center_x = processed_image.shape[1] // 2
    quarter_width = processed_image.shape[1] // 4


    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            # Calculate the bounding rectangle to check the contour's position
            x, y, w, h = cv2.boundingRect(cnt)
            if (x > img_center_x - quarter_width) and (x + w < img_center_x + quarter_width):
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
   
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)

    res = cv2.bitwise_and(processed_image,mask)

    cv2.imshow('Capture', res)
    cv2.waitKey(300)  # Wait for a key press
    cv2.destroyAllWindows()

    return res

##
## Finding horizantal and vertical lines
##


def find_vert_lines(mask_image):
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))

    dx = cv2.Sobel(mask_image,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    #cv2.imshow('Vert', closex)
    #cv2.waitKey(700)  # Wait for a key press
    #cv2.destroyAllWindows()

    return closex

def find_horiz_lines(mask_image):

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(mask_image,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 8:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    closey = close.copy()

    #cv2.imshow('Horiz', closey)
    #cv2.waitKey(700)  # Wait for a key press
    #cv2.destroyAllWindows()

    return closey

def process_centroids_and_label(res, img, res2):
    contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)

        if mom['m00'] != 0:
            x = int(mom['m10']/mom['m00'])
            y = int(mom['m01']/mom['m00'])
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            centroids.append((x, y))
        else:
            print("Skipped a contour with zero area")


    # Assuming 'img' is the original image and 'centroids' contains the centroid points

    # Convert the image to color if it's not already
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print(centroids)

    # Draw the centroids on the image
    for (x, y) in centroids:
        # Draw a small circle at each centroid location
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dot

    # Show the image with the plotted centroids
    #cv2.imshow('Centroids', img)
    #cv2.waitKey(700)  # Wait for a key press
    #cv2.destroyAllWindows()


    ## Correction with centroids

    # Make sure centroids are sorted and reshaped properly
    centroids = np.array(centroids, dtype=np.float32) 

    if len(centroids)!=16:
        print("Problem in detection, re-running program")
        return None, None, None
        
    c = centroids.reshape((16, 2))
    c2 = c[np.argsort(c[:, 1])]

    # Fix the sorting logic for a 4x4 grid (16 points)
    b = np.vstack([c2[i*4:(i+1)*4][np.argsort(c2[i*4:(i+1)*4, 0])] for i in range(4)])
    bm = b.reshape((4, 4, 2)) 

    # Make a copy of the image
    labeled_in_order = res2.copy()

    # Define colors for text and points
    textcolor = (0, 255, 0)  # Green for text
    pointcolor = (255, 0, 0)  # Red for points

    # Iterate over each point to label it
    for index, pt in enumerate(b):
        # Convert point coordinates to integers
        pt = (int(pt[0]), int(pt[1]))
        
        # Label the point on the image
        cv2.putText(labeled_in_order, str(index), pt, cv2.FONT_HERSHEY_DUPLEX, 0.75, textcolor)
        cv2.circle(labeled_in_order, pt, 5, pointcolor, -1)  # -1 to fill the circle

    # Create a named window (to ensure compatibility with different OS)
    winname = "Labeled in Order"
    cv2.namedWindow(winname)

    # Show the labeled image
    cv2.imshow(winname, labeled_in_order)
    cv2.waitKey(700)  # Wait for a key press
    cv2.destroyAllWindows()
    
    return labeled_in_order, b, bm


def process_cells(res2, b, bm):
    output_size = 450  # The size of the output image
    grid_size = 4  # 4x4 grid
    cell_size = output_size // (grid_size-1)

    output = np.zeros((output_size,output_size,3),np.uint8)

    cells = []
    cells_check =[]

    for i,j in enumerate(b):
        ri = int(i/grid_size) # row index
        ci = i%grid_size # column index
        if ci != (grid_size-1) and ri!=(grid_size-1):
            src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
            dst = np.array( [ [ci*cell_size,ri*cell_size],[(ci+1)*cell_size-1,ri*cell_size],[ci*cell_size,(ri+1)*cell_size-1],[(ci+1)*cell_size-1,(ri+1)*cell_size-1] ], np.float32)
            retval = cv2. getPerspectiveTransform(src,dst)
            warp = cv2.warpPerspective(res2,retval,(output_size,output_size),flags=cv2.INTER_CUBIC)
                                   
            output[ri*cell_size:(ri+1)*cell_size-1 , ci*cell_size:(ci+1)*cell_size-1] = warp[ri*cell_size:(ri+1)*cell_size-1 , ci*cell_size:(ci+1)*cell_size-1].copy()

            # Get the perspective transform and apply it to the pre-processed image
            M = cv2.getPerspectiveTransform(src, dst)
            cell = cv2.warpPerspective(res2, M, (cell_size, cell_size), flags=cv2.INTER_CUBIC)
            cell = warp[ri*cell_size:(ri+1)*cell_size-1 , ci*cell_size:(ci+1)*cell_size-1].copy()

            margin_percent = 0.12  # 5% margin
            height, width = cell.shape[:2]
            margin_x = int(width * margin_percent / 2)
            margin_y = int(height * margin_percent / 2)
            
            cell = cell[margin_y:-margin_y, margin_x:-margin_x]
    
            
            if cell.ndim == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            threshold_value = 235
            _, binary_image = cv2.threshold(cell, threshold_value, 255, cv2.THRESH_BINARY)

            captured_image_pil = Image.fromarray(binary_image )
            inverted_image = ImageOps.invert(captured_image_pil)
            # Convert the PIL Image back to a NumPy array before displaying
            inverted_image_np = np.array(inverted_image)
            #cv2.imshow('Original Image', inverted_image_np)
            #cv2.waitKey(700)
            #cv2.destroyAllWindows()


            cells_check.append(inverted_image_np)

            cell_tensor = transform(inverted_image_np).unsqueeze(0)  # Add batch dimension
            #cell_tensor_check= transform(inverted_image_np)
            #cv2.imshow('Original Image', cell_tensor_check)
            #cv2.waitKey(700)
            #cv2.destroyAllWindows()
            cells.append(cell_tensor)

    cv2.imshow('Original Image', output)
    cv2.waitKey(700)
    cv2.destroyAllWindows()

    return output, cells, cells_check

##### Classify cells one by one 

def classify_cells(model, cells, density_threshold=0.006):
    
    cells_value=[]
    # Run each cell through the model
    for cell_tensor in cells:
        with torch.no_grad():        
            
            # Convert the tensor to a NumPy array
            cell_numpy = cell_tensor.squeeze().numpy()

            #cv2.imshow('Original Image', cell_numpy)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            # Calculate the density of white pixels
            white_pixels = np.sum(cell_numpy > 0.6)  # Adjust the threshold for white pixel detection
            total_pixels = cell_numpy.size
            density = white_pixels / total_pixels
            print(density)

            # Check if the density of white pixels is less than the threshold
            if density < density_threshold:
                cell_val=-1
                
            else:
                output2 = model(cell_tensor)
                cell_val=torch.argmax(output2[:, :2], dim=1).item()                
            
            cells_value.append(cell_val)
    
    return cells_value


import tkinter as tk
from tkinter import messagebox

######### Getting Cell Values from image (calling sub-functions) #############################

def get_cell_values_from_image(cap):
    captured_image = capture_image_from_webcam(cap)
    res, res2,mask,img = preprocess_image(captured_image)
    res = create_square_mask_image(res,mask)
    closex = find_vert_lines(res)
    closey = find_horiz_lines(res)
    res = cv2.bitwise_and(closex,closey)
    labeled_in_order, b, bm = process_centroids_and_label(res, img, res2)
    if labeled_in_order is None:
        return None
    output, cells, cells_check = process_cells(res2, b, bm)
    cells_value = classify_cells(model, cells)

    return cells_value

#####################################################################
######## Arduino Communication Functions
#####################################################################

import serial
import time
import re

# Function to initialize the serial connection
def init_serial_connection(port="COM5", baud="115200", timeout=2):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  #some time for setup
        return ser
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return None

def display_message(ser, message):
    command = f"LCD:{message}\n"
    send_grbl_command(ser, command)

# Variables to track target positions
current_target_x = None
current_target_y = None
current_target_z = None

def send_grbl_command(ser, command):
    global command_queue
    command_queue.put(command)

    global current_target_x, current_target_y, current_target_z
    # Update the target positions based on the command
    match = re.search(r'X(-?\d+\.?\d*)', command)
    if match:
        current_target_x = float(match.group(1))
    match = re.search(r'Y(-?\d+\.?\d*)', command)
    if match:
        current_target_y = float(match.group(1))
    match = re.search(r'Z(-?\d+\.?\d*)', command)
    if match:
        current_target_z = float(match.group(1))

    # Send the command
    ser.write((command + '\n').encode())
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "ok":
            break
        elif line:
            print("Received:", line)

    if 'G0' in command or 'G1' in command:  # Check position for G0/G1 commands
        print("Waiting for position to be reached...")
        wait_until_position_reached(ser)
        status_tuple = ("Reached-Idle",current_target_x,current_target_y,current_target_z)
        status_queue.put(status_tuple)

def get_current_position(ser):
    
    ser.write(b'?\n')
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith('<'):            
            return line
        time.sleep(0.1)

def wait_until_position_reached(ser):
    global status_queue
    global current_target_x, current_target_y, current_target_z
    while True:
        response = get_current_position(ser)
        match = re.search(r'MPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)', response)
        if match:
            current_x, current_y, current_z = map(float, match.groups())
            if ((current_target_x is None or abs(current_x - current_target_x) < 0.1) and
                (current_target_y is None or abs(current_y - current_target_y) < 0.1) and
                (current_target_z is None or abs(current_z - current_target_z) < 0.1)):
                return
            status_tuple = (response,current_x,current_y,current_z)
            status_queue.put(status_tuple)

        time.sleep(0.5)

##############################################################
# GUI Build
##############################################################
import customtkinter as ctk

def update_claw_angle_label(value=None):
    claw_angle_value_label.configure(text=f"{claw_angle_slider.get() * 90:.2f}°")

def on_click(row, col): 
    pass #Don't need buttons to do anything , for error handling

def update_xyz_labels(value=None):
    x_label_value.configure(text=f"{x_slider.get() :.2f} mm")
    y_label_value.configure(text=f"{y_slider.get() :.2f} mm")
    z_label_value.configure(text=f"{z_slider.get() :.2f} mm")

def update_alert_label(switch, label):
    if switch.get() == 1:  # Assuming 1 is the state where the limit is activated
        label.configure(text="ALERT", fg_color="red")
    else:
        label.configure(text="", fg_color=ctk.CTk().bg_color)

root = ctk.CTk()
root.title("3D Printer Control")
root.geometry("800x700")

# Main layout frames
left_frame = ctk.CTkFrame(root)
left_frame.pack(side="left", fill="both", expand=True)
right_frame = ctk.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True)

# XYZ Coordinates Frame
ctk.CTkLabel(left_frame, text="Coordinate Locations").pack(pady=(5, 0))
xyz_frame = ctk.CTkFrame(left_frame)
xyz_frame.pack(padx=10, pady=10, fill="both", expand=True)

# X Slider
ctk.CTkLabel(xyz_frame, text="X").grid(row=0, column=0, padx=(5, 0), pady=(5, 0))
x_slider = ctk.CTkSlider(xyz_frame, from_=-140, to=70, command=update_xyz_labels)
x_slider.grid(row=0, column=1, padx=5, pady=5)
x_label_value = ctk.CTkLabel(xyz_frame, text="0.00 mm")
x_label_value.grid(row=0, column=2, padx=(0, 5))

# Y Slider
ctk.CTkLabel(xyz_frame, text="Y").grid(row=1, column=0, padx=(5, 0), pady=(5, 0))
y_slider = ctk.CTkSlider(xyz_frame, from_=-15, to=100, command=update_xyz_labels)
y_slider.grid(row=1, column=1, padx=5, pady=5)
y_label_value = ctk.CTkLabel(xyz_frame, text="0.00 mm")
y_label_value.grid(row=1, column=2, padx=(0, 5))

# Z Slider
ctk.CTkLabel(xyz_frame, text="Z").grid(row=2, column=0, padx=(5, 0), pady=(5, 0))
z_slider = ctk.CTkSlider(xyz_frame, from_=-8.5, to=40, command=update_xyz_labels)
z_slider.grid(row=2, column=1, padx=5, pady=5)
z_label_value = ctk.CTkLabel(xyz_frame, text="0.00 mm")
z_label_value.grid(row=2, column=2, padx=(0, 5))

# Servo Controls Frame
ctk.CTkLabel(left_frame, text="Servo Controls").pack(pady=(5, 0))
servo_frame = ctk.CTkFrame(left_frame)
servo_frame.pack(padx=5, pady=5, fill="both", expand=True)

# Claw Angle Controls
ctk.CTkLabel(servo_frame, text="Claw Angle").pack(side="top", padx=5, pady=5)
claw_angle_slider = ctk.CTkSlider(servo_frame, from_=0, to=90, command=update_claw_angle_label)
claw_angle_slider.pack(side="top", fill="x", expand=True, padx=5, pady=5)
claw_angle_value_label = ctk.CTkLabel(servo_frame, text="0.00°")
claw_angle_value_label.pack(side="top", padx=5, pady=5)

# Claw Status Switch
ctk.CTkLabel(servo_frame, text="Claw Status: Open <=> Close").pack(side="top", padx=5, pady=5)
claw_status_switch = ctk.CTkSwitch(servo_frame, text="")
claw_status_switch.pack(side="top", padx=5, pady=5)

# G-code Frame
ctk.CTkLabel(left_frame, text="G-Code and System Status").pack(pady=(5, 0))
gcode_frame = ctk.CTkFrame(left_frame)
gcode_frame.pack(padx=10, pady=10, fill="both", expand=True)

# G-code Next Command
ctk.CTkLabel(gcode_frame, text="Next G-Code:", anchor="w").pack(fill="x", padx=5, pady=(5, 0))

# GRBL System Status
status_label = ctk.CTkLabel(gcode_frame, text="System Status:")
status_label.pack(fill="x", padx=5, pady=(5, 0))

# Dashboard Frame
ctk.CTkLabel(right_frame, text="Dashboard").pack(pady=(5, 0))
dashboard_frame = ctk.CTkFrame(right_frame)
dashboard_frame.pack(padx=10, pady=10, fill="both", expand=True)

# Addding Tic Tac Toe Dashboard 
tic_tac_toe_buttons = {}

for i in range(3):
    for j in range(3):
        button = ctk.CTkButton(dashboard_frame, text="", width=40, height=40, command=lambda row=i, col=j: on_click(row, col))
        button.grid(row=i, column=j, padx=5, pady=5)
        tic_tac_toe_buttons[(i, j)] = button  # Storing button with grid position

# Operation Controls Frame
ctk.CTkLabel(right_frame, text="Operation Controls").pack(pady=(5, 0))
operation_frame = ctk.CTkFrame(right_frame)
operation_frame.pack(padx=10, pady=10, fill="both", expand=True)

# Limit Switches Status
for i, axis in enumerate(['X', 'Y', 'Z']):
    limit_switch_label = ctk.CTkLabel(operation_frame, text=f"{axis} Limit:")
    limit_switch_label.grid(row=i, column=0, padx=5, pady=5)
    limit_switch_status = ctk.CTkSwitch(operation_frame,text="")
    limit_switch_status.grid(row=i, column=1, padx=5, pady=5)
    # Text-based Alert Indicator
    alert_label = ctk.CTkLabel(operation_frame, text="ALERT", fg_color="red")
    alert_label.grid(row=i, column=2, padx=5, pady=5)

# Control Buttons
for i, button_text in enumerate(["Homing Operation", "Stop", "Manual Control"]):
    button = ctk.CTkButton(operation_frame, text=button_text)
    button.grid(row=3+i, column=0, columnspan=3, padx=5, pady=5)

# Arduino Communication Dashboard Frame
ctk.CTkLabel(right_frame, text="Arduino Communication Dashboard").pack(pady=(5, 0))
dashboard_frame = ctk.CTkFrame(right_frame)
dashboard_frame.pack(padx=10, pady=10, fill="both", expand=True)



########### GUI Build ends here ##################################


######################################################
#Let's play code
######################################################

def translate_values_to_board(cells_value):
    """
    #Translate cell values (-1, 0, 1) to Tic-Tac-Toe board values ("", "X", "O").
    """
    translation = {0: "O", 1: "X", -1: ""}
    return [[translation[cells_value[i * 3 + j]] for j in range(3)] for i in range(3)]

def is_board_full(board):
    for row in board:
        for cell in row:
            if cell == "":
                return False
    return True

def is_winner(board, player):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] == player:
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True 
    return False


def minimax(board, depth, is_maximizing):
    if is_winner(board, "X"):  # AI is 'X'
        return 10 - depth
    if is_winner(board, "O"):  # Human is 'O'
        return depth - 10
    if is_board_full(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ""
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ""
                    best_score = min(score, best_score)
        return best_score

def find_best_move(board):
    best_score = -float('inf')
    move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == "":
                board[i][j] = "X"  # AI makes move
                score = minimax(board, 0, False)
                board[i][j] = ""  # Undo move
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def play_game(cells_value):
    board = translate_values_to_board(cells_value)
    print("Current Board State:")
    print_board(board)

    if is_winner(board, "O"):
        print("Human (O) Wins!")
        return
    if is_board_full(board):
        print("It's a Draw!")
        return

    ai_move = find_best_move(board)
    if ai_move != (-1, -1):
        print(f"AI's Next Move: Place 'X' at Row {ai_move[0]+1}, Column {ai_move[1]+1}")
    else:
        print("No moves left!")

    return ai_move[0]+1, ai_move[1]+1


def draw_line(ser, x, y):
    display_message(ser, "My turn")
    # Commands to move the pen and draw the line
    send_grbl_command(ser, f"G0 X{x} Y{y+6.5}")    # Position pen
    send_grbl_command(ser, "G1 Z-8.9 F300")             # Lower pen
    send_grbl_command(ser, f"G1 Y{y-6.5} F500")         # Draw line
    send_grbl_command(ser, "G0 Z0")                # Lift pen
    send_grbl_command(ser, "G0 X0 Y0")             # Return to home
    time.sleep(2)
    display_message(ser, "Your turn.Press button to play")


def find_winning_cells(board, player):
    
    winning_cells = None

    # Check rows for winner
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] == player:
            winning_cells = [(row, 0), (row, 1), (row, 2)]
            break  # Exit the loop if a winning condition is found
    
    # Check columns for winner, only if no winner found yet
    if winning_cells is None:
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] == player:
                winning_cells = [(0, col), (1, col), (2, col)]
                break  # Exit the loop if a winning condition is found

    # Check diagonals for winner, only if no winner found yet
    if winning_cells is None:
        if board[0][0] == board[1][1] == board[2][2] == player:
            winning_cells = [(0, 0), (1, 1), (2, 2)]
        elif board[0][2] == board[1][1] == board[2][0] == player:
            winning_cells = [(0, 2), (1, 1), (2, 0)]

    print(winning_cells)

    return winning_cells


def draw_winning_line(ser, winning_cells):
    if winning_cells:
        start_x, start_y = get_coordinates_from_move((winning_cells[0][0] + 1, winning_cells[0][1] + 1))
        end_x, end_y = get_coordinates_from_move((winning_cells[-1][0] + 1, winning_cells[-1][1] + 1))
        # Draw line from start to end
        send_grbl_command(ser, f"G0 X{end_x} Y{end_y} Z0")  # Position pen
        send_grbl_command(ser, "G1 Z-8.6 F400")   
        send_grbl_command(ser, f"G1 X{start_x} Y{start_y} F200")            # Draw line start_x start_y
        send_grbl_command(ser, "G0 Z0")                            # Lift pen
        send_grbl_command(ser, "G0 X0 Y0")   

def get_coordinates_from_move(move):
    # Convert the row and column to physical coordinates
    row_offsets = [13, 39, 65]  # Y-coordinates for each row
    col_offsets = [-105, -79, -53]  # X-coordinates for each column
    x = col_offsets[move[1] - 1]
    y = row_offsets[move[0] - 1]
    return x, y

def make_ai_move(ser, move):
    # Convert move to physical coordinates and send command to Arduino
    x, y = get_coordinates_from_move(move)
    draw_line(ser, x, y)

def main_game_loop(ser,cap):
       
    game_over = False
    display_message(ser, "Let's play.Press button to play")

    while not game_over:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"Received from Arduino: {line}")  # Debug print

            if line == "Button: Clicked":
                current_board = get_cell_values_from_image(cap)
                
                if current_board is None:
                    # Handle the error, maybe retry or prompt for a new image capture
                    current_board = get_cell_values_from_image(cap)
                    #continue
                print(current_board)
                board_queue.put(current_board)
                
                board_state = translate_values_to_board(current_board)

                if is_winner(board_state, "O"):
                    display_message(ser, "You Win!")
                    winning_cells = find_winning_cells(board_state, "O")
                    draw_winning_line(ser, winning_cells)
                    game_over = True
                elif is_winner(board_state, "X"):
                    display_message(ser, "I Win!")
                    winning_cells = find_winning_cells(board_state, "X")
                    draw_winning_line(ser, winning_cells)
                    game_over = True
                elif is_board_full(board_state):
                    display_message(ser, "It's a Draw!")
                    game_over = True
                else:
                    display_message(ser, "Calculating move")
                    ai_row, ai_col = play_game(current_board)
                    make_ai_move(ser, (ai_row, ai_col))

                    move_pos_board = (ai_row-1)*3+ ai_col-1
                    current_board[move_pos_board] = 1
                    
                    #Will be handy to verify board incase pen doesn't work                    
                    '''current_board = get_cell_values_from_image() 
                    if current_board is None:
                        # Handle the error, maybe retry or prompt for a new image capture
                        current_board = get_cell_values_from_image()
                        continue
                    #current_board = get_cell_values_from_image()'''

                    board_state = translate_values_to_board(current_board)
                    board_queue.put(current_board)
                    #print(board_state)


                    if is_winner(board_state, "X"):
                        display_message(ser, "I Win!")
                        winning_cells = find_winning_cells(board_state, "X")
                        draw_winning_line(ser, winning_cells)
                        display_message(ser, "I Win!")
                        game_over = True
    

        


#################### Queueing configuration ###################################

######## Initializing global variable########################################

# Create a global queue for inter-thread communication
command_queue = queue.Queue()
board_queue = queue.Queue()
status_queue = queue.Queue()


def cleanup_resources(cap, ser):
    if cap.isOpened():
        cap.release()
    if ser and ser.is_open:
        ser.close()

dashboard_labels = [] 

def update_dashboard2(new_text):
    # Create a new label with the new text
    new_label = ctk.CTkLabel(dashboard_frame, text=new_text)
    new_label.pack(anchor='nw')  # Pack the new label at the top

    # Insert the new label at the beginning of the list
    dashboard_labels.insert(0, new_label)

    # Keep only the latest 5 labels, remove the oldest if necessary
    if len(dashboard_labels) > 5:
        oldest_label = dashboard_labels.pop()
        oldest_label.destroy()

def update_dashboard_arduino_comm():
    
    if not command_queue.empty():
        command = command_queue.get()
        # Update the dashboard with the command
        update_dashboard2(f"Command sent to Arduino: {command}")                               
             
    root.after(100, update_dashboard_arduino_comm)


# Update function
def update_tic_tac_toe_board():
    board_symbols = {0: "O", 1: "X", -1: ""}
    if not board_queue.empty():
        board_state = board_queue.get()
        print(board_state)
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                symbol = board_symbols[board_state[index]]
                tic_tac_toe_buttons[(i, j)].configure(text=symbol)  # Update the button text

    root.after(300, update_tic_tac_toe_board)



def update_grbl_status():
    if not status_queue.empty():
        try:
            status, x_cur, y_cur, z_cur = status_queue.get()
            # Ensure values are not None and are numeric
            if all(isinstance(val, (int, float)) for val in [x_cur, y_cur, z_cur]):
                x_slider.set(x_cur)
                y_slider.set(y_cur)
                z_slider.set(z_cur)
                update_xyz_labels()  # Update labels manually
            else:
                print("Received invalid data:", x_cur, y_cur, z_cur)

            # Update status label
            status_label_text = "System Status: " + status
            status_label.configure(text=status_label_text)

        except Exception as e:
            print("Error updating GRBL status:", e) #Print exception

    root.after(100, update_grbl_status)

def run_game_logic():
    
    try:
        ser = init_serial_connection()
        display_message(ser, "Starting camera")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        time.sleep(1)
        main_game_loop(ser,cap)        
    finally:
        cleanup_resources(cap,ser)
    

# Calling functions to keep updating dashboard
update_dashboard_arduino_comm()
update_tic_tac_toe_board()
update_grbl_status()

# Starting game logic in separate thread
game_thread = threading.Thread(target=run_game_logic)
game_thread.start()

root.mainloop()

