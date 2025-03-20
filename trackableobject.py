class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False
        self.frames_crossed = 0  # Số lần vượt qua đường
        self.disappeared_frames = 0  # Số frame bị mất trước khi loại bỏ
    
    def update(self, new_centroid):
        """ Cập nhật vị trí của đối tượng """
        self.centroids.append(new_centroid)
        self.disappeared_frames = 0  # Reset nếu tìm thấy lại đối tượng
    
    def mark_disappeared(self):
        """ Đánh dấu nếu đối tượng bị mất """
        self.disappeared_frames += 1

    def is_lost(self, max_disappeared=50):
        """ Kiểm tra nếu đối tượng bị mất quá lâu """
        return self.disappeared_frames > max_disappeared

