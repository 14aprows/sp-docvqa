from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_ocr_boxes(
    image,
    words,
    color="gray",
    width=1
):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    for word in words:
        box = [round(value) for value in word["box"]]
        draw.rectangle(
            box,
            outline=color,
            width=width
        )

    return output

def draw_answer_box(
    image,
    answer_box,
    answer_text=None,
    color="red",
    width=5
):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    if answer_box is None:
        return output

    box = [round(value) for value in answer_box]
    draw.rectangle(
        box,
        outline=color,
        width=width
    )
    if answer_text:
        label_position = (box[0], max(0, box[1] - 20))
        draw.text(
            label_position,
            answer_text,
            fill=color
        )

    return output

def draw_answer_words(
    image,
    matched_words,
    word_color="blue",
    answer_color="red"
):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    if not matched_words:
        return output

    for word in matched_words:
        box = [round(value) for value in word["box"]]
        draw.rectangle(
            box,
            outline=word_color,
            width=3
        )

    merged_box = [
        min(word["box"][0] for word in matched_words),
        min(word["box"][1] for word in matched_words),
        max(word["box"][2] for word in matched_words),
        max(word["box"][3] for word in matched_words)
    ]

    draw.rectangle(
        [
            round(value)
            for value in merged_box
        ],
        outline=answer_color,
        width=5
    )

    return output

def denormalize_box(
    box: List[int],
    image_width: int,
    image_height: int
) -> List[int]:
    if len(box) != 4:
        raise ValueError("Bounding box harus memiliki 4 koordinat")
    x_min = round(box[0] / 1000 * image_width)
    y_min = round(box[1] / 1000 * image_height)
    x_max = round(box[2] / 1000 * image_width)
    y_max = round(box[3] / 1000 * image_height)

    return [
        max(0, min(x_min, image_width)),
        max(0, min(y_min, image_height)),
        max(0, min(x_max, image_width)),
        max(0, min(y_max, image_height))
    ]

def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    pixel_box: List[int],
    label: str,
    color: Tuple[int, int, int],
    width: int = 4
):
    x_min, y_min, x_max, y_max = pixel_box

    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline=color,
        width=width,
    )

    font = ImageFont.load_default()
    text_bbox = draw.textbbox(
        (x_min, y_min),
        label,
        font=font,
    )

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_top = max(0, y_min - text_height - 6)

    draw.rectangle(
        [x_min, label_top, x_min + text_width + 8, label_top + text_height + 6],
        fill=color,
    )

    draw.text(
        (x_min + 4, label_top + 3),
        label,
        fill=(255, 255, 255),
        font=font,
    )