# import cv2
# import numpy as np
# from filterpy.kalman import KalmanFilter

# class Tracker:
#     def __init__(self, max_distance=70, max_disappeared=30, min_box_size=40):
#         self.tracked_objects = {}  # Lưu danh sách object đang theo dõi
#         self.lost_objects = {}  # Lưu object bị mất tạm thời
#         self.kalman_filters = {}  # Bộ lọc Kalman
#         self.max_distance = max_distance
#         self.max_disappeared = max_disappeared
#         self.min_box_size = min_box_size
#         self.next_object_id = 1  # ID object tiếp theo

#     def init_kalman_filter(self, obj_id, bbox):
#         """Khởi tạo bộ lọc Kalman để dự đoán vị trí khi bị che lấp"""
#         kf = KalmanFilter(dim_x=4, dim_z=2)
#         kf.x = np.array([bbox[0], bbox[1], 0, 0])
#         kf.F = np.array([[1, 0, 1, 0],
#                          [0, 1, 0, 1],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]])
#         kf.P *= 1000
#         self.kalman_filters[obj_id] = kf

#     def update(self, new_rectangles):
#         updated_objects = {}

#         # Lọc bỏ bounding box quá nhỏ
#         new_rectangles = [rect for rect in new_rectangles if (rect[2] - rect[0] >= self.min_box_size and rect[3] - rect[1] >= self.min_box_size)]

#         for new_rect in new_rectangles:
#             matched = False

#             # Kiểm tra với object đang track
#             for obj_id, obj_rect in self.tracked_objects.items():
#                 new_center = ((new_rect[0] + new_rect[2]) / 2, (new_rect[1] + new_rect[3]) / 2)
#                 obj_center = ((obj_rect[0] + obj_rect[2]) / 2, (obj_rect[1] + obj_rect[3]) / 2)

#                 distance = ((new_center[0] - obj_center[0]) ** 2 + (new_center[1] - obj_center[1]) ** 2) ** 0.5

#                 if distance <= self.max_distance:
#                     updated_objects[obj_id] = new_rect
#                     matched = True
#                     break

#             if not matched:
#                 for obj_id, disappeared_frames in list(self.lost_objects.items()):
#                     if disappeared_frames < self.max_disappeared:
#                         updated_objects[obj_id] = new_rect
#                         del self.lost_objects[obj_id]
#                         matched = True
#                         break

#             if not matched:
#                 updated_objects[self.next_object_id] = new_rect
#                 self.init_kalman_filter(self.next_object_id, new_rect)
#                 self.next_object_id += 1

#         disappeared_objects = set(self.tracked_objects.keys()) - set(updated_objects.keys())
#         for obj_id in disappeared_objects:
#             self.lost_objects[obj_id] = self.lost_objects.get(obj_id, 0) + 1
#             if self.lost_objects[obj_id] >= self.max_disappeared:
#                 del self.lost_objects[obj_id]

#         self.tracked_objects = updated_objects
#         return self.tracked_objects


import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

class Tracker:
    def __init__(self, max_distance=70, max_disappeared=30, min_box_size=40):
        self.tracked_objects = {}  # Lưu danh sách object đang theo dõi
        self.lost_objects = {}  # Lưu object bị mất tạm thời
        self.kalman_filters = {}  # Bộ lọc Kalman
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_box_size = min_box_size
        self.next_object_id = 1  # ID object tiếp theo

    def init_kalman_filter(self, obj_id, bbox):
        """Khởi tạo bộ lọc Kalman để dự đoán vị trí khi bị che lấp"""
        kf = KalmanFilter(dim_x=6, dim_z=2)  # Sử dụng 6 chiều (x, y, vx, vy, ax, ay)
        kf.x = np.array([bbox[0], bbox[1], 0, 0, 0, 0])  # Khởi tạo trạng thái ban đầu
        kf.F = np.array([[1, 0, 1, 0, 0.5, 0],  # Ma trận chuyển đổi trạng thái
                         [0, 1, 0, 1, 0, 0.5],
                         [0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],  # Ma trận đo lường
                         [0, 1, 0, 0, 0, 0]])
        kf.P *= 1000  # Ma trận hiệp phương sai ban đầu
        kf.R = np.eye(2) * 10  # Nhiễu đo lường
        kf.Q = np.eye(6) * 0.01  # Nhiễu quá trình
        self.kalman_filters[obj_id] = kf

    def update(self, new_rectangles):
        updated_objects = {}

        # Lọc bỏ bounding box quá nhỏ
        new_rectangles = [rect for rect in new_rectangles if (rect[2] - rect[0] >= self.min_box_size and rect[3] - rect[1] >= self.min_box_size)]

        for new_rect in new_rectangles:
            matched = False

            # Kiểm tra với object đang track
            for obj_id, obj_rect in self.tracked_objects.items():
                new_center = ((new_rect[0] + new_rect[2]) / 2, (new_rect[1] + new_rect[3]) / 2)
                obj_center = ((obj_rect[0] + obj_rect[2]) / 2, (obj_rect[1] + obj_rect[3]) / 2)

                distance = ((new_center[0] - obj_center[0]) ** 2 + (new_center[1] - obj_center[1]) ** 2) ** 0.5

                if distance <= self.max_distance:
                    updated_objects[obj_id] = new_rect
                    matched = True
                    break

            if not matched:
                for obj_id, disappeared_frames in list(self.lost_objects.items()):
                    if disappeared_frames < self.max_disappeared:
                        updated_objects[obj_id] = new_rect
                        del self.lost_objects[obj_id]
                        matched = True
                        break

            if not matched:
                updated_objects[self.next_object_id] = new_rect
                self.init_kalman_filter(self.next_object_id, new_rect)
                self.next_object_id += 1

        disappeared_objects = set(self.tracked_objects.keys()) - set(updated_objects.keys())
        for obj_id in disappeared_objects:
            self.lost_objects[obj_id] = self.lost_objects.get(obj_id, 0) + 1
            if self.lost_objects[obj_id] >= self.max_disappeared:
                del self.lost_objects[obj_id]
                if obj_id in self.kalman_filters:
                    del self.kalman_filters[obj_id]

        self.tracked_objects = updated_objects
        return self.tracked_objects