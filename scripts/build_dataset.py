import pandas as pd
import os
import cairosvg
import subprocess
from PIL import Image
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

    print("Processing images and creating dataset pairs...")
    dataset_records = []
    output_dir = "data/processed_images"
    os.makedirs(output_dir, exist_ok=True)

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
        png_path = os.path.join(output_dir, f"{hexcode}.png")
        if os.path.exists(svg_path):
            temp_png_path = png_path + ".tmp"
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
            img = Image.open(temp_png_path)
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img.save(png_path)
            os.remove(temp_png_path)

            base_annotation = annotation.strip()
            dataset_records.append({"image_path": png_path, "caption": base_annotation})

            if tags:
                caption_keywords = f"{base_annotation}, a symbol of {tags}"
                dataset_records.append(
                    {"image_path": png_path, "caption": caption_keywords}
                )

            caption_hierarchy = (
                f"An emoji from the {group} category: {base_annotation}"
            )
            dataset_records.append(
                {"image_path": png_path, "caption": caption_hierarchy}
            )

            if subgroup:
                caption_hierarchy_full = f"An emoji from the {group} category and the {subgroup} subgroup: {base_annotation}"
                dataset_records.append(
                    {"image_path": png_path, "caption": caption_hierarchy_full}
                )
                caption_subgroup = f"{base_annotation}, belonging to the {subgroup} subgroup"
                dataset_records.append(
                    {"image_path": png_path, "caption": caption_subgroup}
                )

    print("Saving the final dataset to pairs.csv...")
    final_df = pd.DataFrame(dataset_records)
    final_df.to_csv("data/pairs.csv", index=False)
    print("Dataset build complete!")


if __name__ == "__main__":
    build_dataset()
