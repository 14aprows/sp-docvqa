def merge_boxes(boxes):
    if not boxes:
        return None

    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes)
    ]

def normalize_box(
    box,
    image_width,
    image_height,
    target_size=1000
):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Ukuran gambar harus lebih besar dari 0")

    x1, y1, x2, y2 = box
    normalized = [
        round(target_size * x1 / image_width),
        round(target_size * y1 / image_height),
        round(target_size * x2 / image_width),
        round(target_size * y2 / image_height)
    ]

    return [max(0, min(target_size, value)) for value in normalized]