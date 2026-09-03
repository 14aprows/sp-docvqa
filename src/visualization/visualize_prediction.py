from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from src.visualization.draw_bbox import denormalize_box, draw_labeled_box

def visualize_prediction(
    image_path: str,
    predicted_bbox: List[int],
    predicted_answer: str,
    question: str,
    output_path: str,
    ground_truth_bbox: Optional[List[int]] = None,
    ground_truth_answer: Optional[str] = None
):
    source_path = Path(image_path)
    destination_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as source_image:
        document_image = source_image.convert("RGB")

    image_width, image_height = document_image.size
    draw = ImageDraw.Draw(document_image)

    if ground_truth_bbox is not None and ground_truth_bbox != [0, 0, 0, 0]:
        ground_truth_pixel_box = denormalize_box(
            ground_truth_bbox,
            image_width,
            image_height
        )
        draw_labeled_box(
            draw=draw,
            pixel_box=ground_truth_pixel_box,
            label="GROUND TRUTH",
            color=(20, 150, 60),
            width=5
        )

    if predicted_bbox != [0, 0, 0, 0]:
        predicted_pixel_box = denormalize_box(
            predicted_bbox,
            image_width,
            image_height
        )
        draw_labeled_box(
            draw=draw,
            pixel_box=predicted_pixel_box,
            label="PREDICTION",
            color=(220, 40, 40),
            width=3
        )

    header_height = 95
    canvas = Image.new(
        mode="RGB",
        size=(image_width, image_height + header_height),
        color=(255, 255, 255)
    )
    canvas.paste(document_image, (0, header_height))

    header_draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header_draw.text(
        (10, 10),
        f"Question: {question}",
        fill=(0, 0, 0),
        font=font
    )
    header_draw.text(
        (10, 35),
        f"Prediction: {predicted_answer}",
        fill=(220, 40, 40),
        font=font
    )

    if ground_truth_answer is not None:
        header_draw.text(
            (10, 60),
            f"Ground truth: {ground_truth_answer}",
            fill=(20, 130, 60),
            font=font
        )

    canvas.save(destination_path)