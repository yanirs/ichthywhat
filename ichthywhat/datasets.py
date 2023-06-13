"""Dataset handling functionality."""

import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


def create_rls_genus_dataset(
    *, image_dir: Path, output_dir: Path, num_top_genera: int
) -> None:
    """Create a dataset of the top genera from an RLS image directory.

    Images are cropped to remove the RLS URL.

    Parameters
    ----------
    image_dir
        Path of an RLS image directory, containing files named
        `<genus>-<taxon>-<num>.<extension>`.
    output_dir
        Path of the output directory, which must not exist.
    num_top_genera
        Number of top genera to include, where ranking is based on the number of images
        per genus.
    """
    raise NotImplementedError(
        "Need to handle duplicates as in create_rls_species_dataset()"
    )
    genus_to_num_images = Counter()
    for image_path in image_dir.iterdir():
        genus = image_path.name.split("-", maxsplit=1)[0]
        genus_to_num_images[genus] += 1
    output_dir.mkdir(parents=True)
    print(
        f"Found {len(genus_to_num_images)} genera. "
        f"Copying top {num_top_genera} to {output_dir}"
    )
    for genus, _ in genus_to_num_images.most_common(num_top_genera):
        for src_filename in image_dir.glob(f"{genus}-*"):
            _crop_image_file(src_filename, output_dir / src_filename.name)


def create_rls_species_dataset(
    *,
    m1_csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    num_species: int | None = None,
    min_images_per_species: int = 1,
) -> None:
    """Create a dataset directory from an RLS image directory.

    Images are cropped to remove the RLS URL.

    Parameters
    ----------
    m1_csv_path
        Path of an M1 CSV file, as downloaded from RLS.
    image_dir
        Path of an RLS image directory, containing files named
        `<genus>-<taxon>-<num>.<extension>`.
    output_dir
        Path of the output directory, which must not exist.
    num_species
        Number of species to include in the dataset. If None, all species are included.
    min_images_per_species
        Only species with at least this number of images will be included.
    """
    species_with_min_images = set()
    species_with_duplicates = set()
    hash_to_image_path: dict[int, Path] = {}
    for image_path in image_dir.iterdir():
        try:
            genus, taxon, suffix = image_path.name.split("-")
        except ValueError:
            print(f"Skipping {image_path.name} (bad name)")
            continue
        image_hash = hash(image_path.read_bytes())
        if image_hash in hash_to_image_path:
            species_with_duplicates.add(
                " ".join(
                    hash_to_image_path[image_hash].name.split("-")[:2]
                ).capitalize()
            )
            species_with_duplicates.add(f"{genus.capitalize()} {taxon}")
            print(
                f"Skipping {image_path.name} "
                f"(duplicate of {hash_to_image_path[image_hash].name})"
            )
            continue
        hash_to_image_path[image_hash] = image_path
        image_index = int(suffix.split(".")[0])
        if image_index >= min_images_per_species - 1:
            species_with_min_images.add(f"{genus.capitalize()} {taxon}")
    print(f"Ignoring species with duplicates: {sorted(species_with_duplicates)}")
    species_with_min_images.difference_update(species_with_duplicates)
    m1_species = pd.read_csv(m1_csv_path)["Taxon"].drop_duplicates()
    sampled_species = m1_species[m1_species.isin(species_with_min_images)]
    print(
        f"Found {len(sampled_species)} M1 species with at least "
        f"{min_images_per_species} images"
    )
    if num_species:
        sampled_species = sampled_species.sample(num_species, random_state=0)
    output_dir.mkdir(parents=True)
    print(f"Copying images for {len(sampled_species)} species to {output_dir}")
    for image_glob in sampled_species.str.replace(" ", "-").str.lower() + "-*":
        for src_filename in image_dir.glob(image_glob):
            _crop_image_file(src_filename, output_dir / src_filename.name)


def _crop_image_file(src: Path, dst: Path, top_bottom_pixels: int = 55) -> None:
    """Crop the top and bottom of an image.

    The default pixel count is useful for removing the RLS URL.
    """
    with Image.open(src) as im:
        width, height = im.size
        im.crop((0, top_bottom_pixels, width, height - top_bottom_pixels)).save(dst)


def create_test_dataset(*, trip_dir: Path, output_dir: Path) -> None:
    """Create a test dataset from a trip directory, which is traversed recursively.

    This function attempts to filter out unlabelled files (e.g., those that include
    `sp`). The mapping from the original filenames is written to `src_to_dst.csv` in the
    output directory.

    Parameters
    ----------
    trip_dir
        Path of a trip directory, containing files named
        `<ID> - [species][ AND <species>]*.<jpg|JPG>`.
    output_dir
        Path of the output directory, which must not exist.
    """
    output_dir.mkdir(parents=True)
    src_to_dst = {}
    for child_path in trip_dir.glob("**/* - *.[Jj][Pp][Gg]"):
        img_id, raw_species = child_path.name[:-4].split(" - ", maxsplit=1)
        species_to_keep = []
        for name in raw_species.lower().split(" and "):
            try:
                genus, taxon = name.split(" ")
            except ValueError:
                continue
            if "[" in name or "(" in name or taxon == "todo" or taxon == "sp":
                continue
            species_to_keep.append(f"{genus}-{taxon}")
        dst = (
            f"{img_id}-{'-AND-'.join(species_to_keep)}.jpg" if species_to_keep else None
        )
        if dst:
            shutil.copy(child_path, output_dir / dst)
        src_to_dst[child_path] = dst
    pd.Series(src_to_dst).to_csv(output_dir / "src_to_dst.csv", header=False)
    print(
        f"Copied {len(set(filter(None, src_to_dst.values())))} images. "
        f"See {output_dir / 'src_to_dst.csv'} for details"
    )
