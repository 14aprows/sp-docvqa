from typing import List

def union_boxes(boxes: List[List[int]]) -> List[int]:
    if not boxes:
        return [0, 0, 0 , 0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes)
    ]

def bbox_iou(first_box: List[int], second_box: List[int]) -> float:
    intersection_left = max(first_box[0], second_box[0])
    intersection_top = max(first_box[1], second_box[1])
    intersection_right = min(first_box[2], second_box[2])
    intersection_bottom = min(first_box[3], second_box[3])

    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    first_area = max(0, first_box[2] - first_box[0]) * max(0, first_box[3] - first_box[1])
    second_area = max(0, second_box[2] - second_box[0]) * max(0, second_box[3] - second_box[1])

    union_area = first_area + second_area - intersection_area
    if union_area == 0:
        return 0.0

    return intersection_area / union_area