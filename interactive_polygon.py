import cv2
import pandas as pd
import numpy as np
import os
import shutil

points = []  # 存储顶点
polygon_counter = 0  # 多边形计数器
buttons = {
    "reset": {"pos": (10, 10), "size": (60, 30), "label": "Reset"},
    "save": {"pos": (80, 10), "size": (60, 30), "label": "Save"},
    "close": {"pos": (150, 10), "size": (60, 30), "label": "Close"}
}
scale = 0.7  # 缩放比例

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)

def draw_buttons(image):
    for key, button in buttons.items():
        x, y = button["pos"]
        w, h = button["size"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, button["label"], (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def check_button_click(x, y):
    for key, button in buttons.items():
        bx, by = button["pos"]
        bw, bh = button["size"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return key
    return None

def mouse_click(event, x, y, flags, param):
    global points, img, polygon_counter
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        button_clicked = check_button_click(x, y)
        if button_clicked:
            if button_clicked == "reset":
                clear_folder(folder_path)  # 清空文件夹
                points.clear()
                img = param.copy()  # Reset to the original image
            elif button_clicked == "save":
                if len(points) > 2:  # 保存前确认至少有三个点以构成多边形
                    polygon_name = f'/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/polygon_counter/polygon_vertices{polygon_counter}.xlsx'
                    pd.DataFrame(points, columns=['X', 'Y']).to_excel(polygon_name, index=False)
                    print(f"Polygon vertices saved to '{polygon_name}'.")
                    polygon_counter += 1  # 更新计数器
                    points.clear()  # 清空点，准备下一个多边形
                else:
                    print("Not enough points to form a polygon.")
            elif button_clicked == "close":
                if len(points) > 2:  # 有足够的点闭合多边形
                    cv2.line(img, points[-1], points[0], (255, 0, 0), 2)  # 闭合多边形
                    cv2.imshow("Polygon Drawer", img)
                    cv2.waitKey(1)  # 短暂等待以显示闭合的多边形
            draw_buttons(img)
        else:
            points.append((x, y))
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
            cv2.circle(img, (x, y), 3,(0, 255, 0), -1)
        cv2.imshow("Polygon Drawer", img)

def draw_polygon(video_path):
    global img
    cap = cv2.VideoCapture(video_path)
    ret, original_img = cap.read()
    if not ret:
        print("Failed to read video")
        return
    # 缩放图像
    img = cv2.resize(original_img, None, fx=scale, fy=scale)
    cv2.namedWindow("Polygon Drawer")
    cv2.setMouseCallback("Polygon Drawer", mouse_click, img)
    draw_buttons(img)

    while True:
        cv2.imshow("Polygon Drawer", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # 按ESC退出
            save_path = '/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/Road_pic/escaped_frame.jpg'
            cv2.imwrite(save_path, img)
            print(f"Frame saved at {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = '/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/polygon_counter'
    clear_folder(folder_path)  # 清空文件夹
    video_path = '/home/chris007/python/PRB_SM/AICITY_2023_Track5-main/AICITY_2023_Track5-main/video/dowload/053101.mp4'  # 视频路径
    draw_polygon(video_path)
