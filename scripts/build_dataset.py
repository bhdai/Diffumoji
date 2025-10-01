import pandas as pd
import os
import cairosvg
import subprocess
from PIL import Image
import random
from tqdm import tqdm


def build_dataset():
    DATA_DIR = "data"
    OPENMOJI_DIR = os.path.join(DATA_DIR, "openmoji")

    if not os.path.exists(OPENMOJI_DIR):
        print("OpenMoji dataset not found. Cloning from github...")
        os.makedirs(OPENMOJI_DIR, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/hfg-gmuend/openmoji.git",
                OPENMOJI_DIR,
            ]
        )
    else:
        print("OpenMoji dataset found.")

    print("Loading metadata from openmoji.csv...")
    metadata_path = os.path.join(OPENMOJI_DIR, "data", "openmoji.csv")
    metadata_df = pd.read_csv(metadata_path)

    output_dir = "data/lora_dataset/10_emojistyle"
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = "data/processed_images"
    os.makedirs(temp_dir, exist_ok=True)

    print("Processing images and creating LoRA dataset...")
    image_counter = 1
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        hexcode = row["hexcode"]
        annotation = row["annotation"]
        tags = row["tags"]
        group = row["group"]
        subgroup = row["subgroups"]

        if not isinstance(annotation, str) or not isinstance(group, str):
            continue
        tags = tags if isinstance(tags, str) else ""
        subgroup = subgroup if isinstance(subgroup, str) else ""

        svg_path = os.path.join(OPENMOJI_DIR, "color", "svg", f"{hexcode}.svg")
        temp_png_path = os.path.join(temp_dir, f"{hexcode}.png")

        if os.path.exists(svg_path):
            try:
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=temp_png_path,
                    output_width=512,
                    output_height=512,
                )
            except Exception as e:
                print(f"Error converting {svg_path}: {e}")
                continue
            img = Image.open(temp_png_path).convert("RGBA")
            # paste onto white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # use alpha as mask
            background = background.resize((64, 64), Image.Resampling.LANCZOS)
            background.save(temp_png_path)

            captions_for_this_image = []
            base_annotation = annotation.strip()
            captions_for_this_image.append(base_annotation)

            if tags:
                captions_for_this_image.append(f"{base_annotation}, a symbol of {tags}")
            captions_for_this_image.append(
                f"An emoji from the {group} category: {base_annotation}"
            )

            if subgroup:
                captions_for_this_image.append(
                    f"An emoji from the {group} category and {subgroup} subgroup: {base_annotation}"
                )
                captions_for_this_image.append(
                    f"{base_annotation}, belonging to the {subgroup} subgroup"
                )

            if not captions_for_this_image:
                os.remove(temp_png_path)
                continue

            chosen_caption = random.choice(captions_for_this_image)
            final_caption = f"emojistyle, {chosen_caption}"

            new_image_path = os.path.join(output_dir, f"{image_counter:04d}.png")
            new_caption_path = os.path.join(output_dir, f"{image_counter:04d}.txt")

            os.rename(temp_png_path, new_image_path)
            with open(new_caption_path, "w") as f:
                f.write(final_caption)
            image_counter += 1
        else:
            print(f"SVG not found: {svg_path}")

    print(
        f"Dataset build complete! Created {image_counter - 1} image-caption pairs in {output_dir}."
    )


if __name__ == "__main__":
    build_dataset()
