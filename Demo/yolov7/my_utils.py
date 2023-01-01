import shutil
import time
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def plot_pose(img, output):
    image = letterbox(img.copy(), 960, stride=64, auto=True)[0]
    for idx in range(output.shape[0]):
      plot_skeleton_kpts(image, output[idx, 7:].T, 3)
      x,y,w,h = output[idx, 2:6]
      if idx ==0:
        cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 1)
    # resize image
    image = cv2.resize(image, (img.shape[1], img.shape[0]))
    return image

def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def test_video(path_video, out_filename, model):
  # change fps of video
  cap = cv2.VideoCapture(path_video)
  n_pose = 0

  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  cap.set(cv2.CAP_PROP_FPS, 10)
  fps = int(cap.get(5))
  video_info = f'frame_width: {frame_width}\nframe_height: {frame_height}\nfps: {fps}'
  print(video_info)

  # Define the codec and create VideoWriter object with fps =20.0
  fps_new = 5
  out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps_new, (frame_width,frame_height))
  count = 0
  id_image = 0
  previous_pose = np.zeros(33)
  active_pose = np.zeros(33)
  while(cap.isOpened()):
      ret, frame = cap.read()
      active = True
      if ret==True:
          if count%(fps//fps_new) == 0:
              output = model.predict(frame)
              frame = plot_pose(frame, output)
              # write the flipped frame
              out.write(frame)
          count += 1
      else:
          break

  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()


def angle_of_4_points(p1, p2, p3, p4):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    v1 = p2 - p1
    v2 = p4 - p3
    angle = np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))
    angle = np.abs(np.degrees(angle))
    if angle > 90:
        angle = 180 - angle
    return angle
  
def angle_of_special_key(points,index_key):
  i,j,k,l = index_key[0], index_key[1], index_key[2], index_key[3]
  x1, y1 = points[7+i*3], points[8+i*3]
  x2, y2 = points[7+j*3], points[8+j*3]
  x3, y3 = points[7+k*3], points[8+k*3]
  x4, y4 = points[7+l*3], points[8+l*3]
  return angle_of_4_points((x1, y1), (x2, y2), (x3, y3), (x4, y4))


def draw_line(img,points,list_angle_key):
    list_image = []
    for i in range(len(list_angle_key)):
        image = img.copy()
        image = letterbox(image, 960, stride=64, auto=True)[0]
        index_key = list_angle_key[i]
        i,j,k,l = index_key[0], index_key[1], index_key[2], index_key[3]
        x1, y1 = points[7+i*3], points[8+i*3]
        x2, y2 = points[7+j*3], points[8+j*3]
        x3, y3 = points[7+k*3], points[8+k*3]
        x4, y4 = points[7+l*3], points[8+l*3]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
        cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 10)
        # ## draw angle
        angle = angle_of_4_points((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        image = cv2.resize(image, img.shape[1::-1], interpolation=cv2.INTER_AREA)
        cv2.putText(image, str(int(angle)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 0), 5)
        list_image.append(image)

    ## plt all 9 image
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for i in range(3):
        for j in range(3):
            # change color of image
            new_iamge = cv2.cvtColor(list_image[i*3+j], cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(new_iamge)
            axs[i, j].set_title(f'angle: {list_angle_key[i*3+j]}')


def angle_vector(img,points,list_angle_key):
    list_angle = []
    for i in range(len(list_angle_key)):
        angle = angle_of_special_key(points,list_angle_key[i])
        list_angle.append(angle)
    return np.array(list_angle)

def encoder_image(img, model, list_angle_key):
  output = model.predict(img)
  encoder_Vector = angle_vector(img, output[0], list_angle_key)
  return encoder_Vector

def take_list_actions(path_folder, model, list_angle_key):
  list_actions = []
  list_names = []
  path_images = glob.glob(path_folder+"/*")
  for i in path_images:
    img = cv2.imread(i)
    action = encoder_image(img, model, list_angle_key)
    list_actions.append(action)
    name = i.split("/")[-1]
    list_names.append(name)
  return np.array(list_actions), list_names

def compare_two_poses(vector1, vector2, mask=np.ones(9)):
  error =  np.sum(np.abs(vector1 - vector2)*mask)
  return error

def compare_with_list_action(vector, action_vector, action_name, action_mask):
  list_error = []
  for i in action_vector:
    list_error.append(compare_two_poses(vector,i,action_mask))
  index = np.argmin(np.array(list_error))
  return list_error[index], action_name[index]

def action_predict_video(path_video, out_filename, model, action_vector, action_name, action_mask, list_angle_key):
  threshold = np.count_nonzero(action_mask)*15
  action_cycle = np.zeros(len(action_vector))
  count_action = 0
  # change fps of video
  cap = cv2.VideoCapture(path_video)
  n_pose = 0

  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  cap.set(cv2.CAP_PROP_FPS, 10)
  fps = int(cap.get(5))
  video_info = f'frame_width: {frame_width}\nframe_height: {frame_height}\nfps: {fps}'
  print(video_info)

  # Define the codec and create VideoWriter object with fps =20.0
  fps_new = 5
  out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps_new, (frame_width,frame_height))
  count_frame = 0
  id_image = 0
  previous_pose = np.zeros(33)
  active_pose = np.zeros(33)
  while(cap.isOpened()):
      ret, frame = cap.read()
      active = True
      if ret==True:
          if count_frame%(fps//fps_new) == 0:
                output = model.predict(frame)
                try:
                    frame = plot_pose(frame, output)
                    frame_endcoder = encoder_image(frame, model, list_angle_key)
                    loss_frame, name_frame = compare_with_list_action(
                    frame_endcoder, action_vector, action_name, action_mask)
                    color = (0, 0, 255)
                    if loss_frame < threshold:
                        color = (0, 255, 0)
                        action_index = action_name.index(name_frame)
                        if np.array(action_cycle[:action_index]).all() == 1:
                            action_cycle[action_index] = 1
                            if np.array(action_cycle).all() == 1:
                                count_action += 1
                                action_cycle = np.zeros(len(action_vector))
                    
                    frame = cv2.putText(frame, name_frame, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)
                    frame = cv2.putText(frame, str(loss_frame), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)
                    frame = cv2.putText(frame, str(count_action), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 10)
                    # write action_cycle
                    frame = cv2.putText(frame, str(action_cycle), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
                except:
                    pass
                # write the flipped frame
                out.write(frame)
          count_frame += 1
      else:
          break

  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()
