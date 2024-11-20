import math

class Tracker:
    def __init__(self):
        self.center_points = {}  # Store the center points of objects
        self.id_count = 0        # ID counter for new objects

    def update(self, objects_rect):
        objects_bbs_ids = []  # List to store tracked objects with IDs

        # Calculate center points for current frame
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if the object is already being tracked
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 25:  # Adjust distance threshold as needed
                    self.center_points[id] = (cx, cy)  # Update center point
                    objects_bbs_ids.append([x1, y1, x2, y2, id])
                    same_object_detected = True
                    break

            # If the object is new, assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Remove IDs not present in the current frame
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()

        return objects_bbs_ids
