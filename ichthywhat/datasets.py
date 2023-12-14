"""Dataset handling functionality."""
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from hashlib import md5
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


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


def create_rls_species_dataset_from_local(
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


def create_rls_species_dataset_from_api(
    *,
    output_dir: Path,
    species_json_url: str = "https://raw.githubusercontent.com/yanirs/rls-data/master/output/species.json",
    methods: Sequence[int] = (1,),
) -> None:
    """Create a dataset directory from the API species.json.

    Images are cropped to remove the RLS URL. Duplicate images within a species are
    ignored, while cross-species duplicates are removed.

    Parameters
    ----------
    output_dir
        Path of the output directory, which must not exist.
    species_json_url
        URL of the species.json, defaulting to the latest rls-data repo version. This
        can be used to regenerate a dataset based on a historical version of the JSON.


    Methods
    -------
        Only species that are counted with these RLS methods are included.
    """
    output_dir.mkdir(parents=True)
    all_species = requests.get(f"{species_json_url}").json()
    hash_to_image_paths = defaultdict(list)
    for species in all_species:
        species_str = (
            f"{species['scientific_name']} "
            f"(https://reeflifesurvey.com/species/{species['slug']}/)"
        )
        if "methods" not in species:
            assert "class" not in species
            print(f"Missing method & class for {species_str}")
            continue
        if not set(species["methods"]).intersection(methods):
            continue
        species_image_hashes = set()
        for photo in species.get("photos", []):
            image_bytes = requests.get(photo["large_url"]).content
            image_hash = md5(image_bytes).hexdigest()
            if image_hash in species_image_hashes:
                print(f"Found duplicate photo for {species_str}")
                continue
            image_path = output_dir / f"{species['slug']}-{image_hash}.jpg"
            try:
                _crop_image_file(BytesIO(image_bytes), image_path)
            except UnidentifiedImageError:
                print(f"Couldn't load {photo['large_url']} for {species_str}")
                continue
            species_image_hashes.add(image_hash)
            hash_to_image_paths[image_hash].append(image_path)
    # Remove images that are reused across species.
    for image_paths in hash_to_image_paths.values():
        if len(image_paths) > 1:
            for image_path in image_paths:
                print(f"Deleting duplicate image: {image_path.name}")
                image_path.unlink()


def _crop_image_file(
    src: Path | BytesIO, dst: Path, top_bottom_pixels: int = 55
) -> None:
    """Crop the top and bottom of an image.

    The default pixel count is useful for removing the RLS URL.
    """
    with Image.open(src) as im:
        width, height = im.size
        im.crop((0, top_bottom_pixels, width, height - top_bottom_pixels)).convert(
            "RGB"
        ).save(dst)


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
