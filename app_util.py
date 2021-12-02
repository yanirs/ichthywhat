"""Utilities to use as part of the streamlit app."""

from collections import defaultdict
import typing

import pandas as pd
import streamlit as st
from geopy.distance import geodesic


@st.experimental_memo
def load_species_df(path_or_url: str, image_root: typing.Optional[str] = None) -> pd.DataFrame:
    """
    Load the species DataFrame from the JSON used for other RLS tools (Frequency Explorer & Flashcards).

    :param path_or_url: local path or URL of the JSON.
    :param image_root: optional path to the root directory of the species images, which will be prepended to each image.

    :return: the species DataFrame.
    """
    species_df = pd.read_json(path_or_url, orient="index")
    species_df.columns = ["name", "common_names", "url", "method", "images"]
    species_df["method"] = species_df["method"].map({0: "M1", 1: "M2", 2: "Both"})
    if image_root:
        species_df["images"] = species_df["images"].map(lambda images: [image_root + image for image in images])
    return species_df


@st.experimental_memo
def load_site_df(path_or_url: str, species_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the site DataFrame from the JSON used for other RLS tools (Frequency Explorer & Flashcards).

    :param path_or_url: local path or URL of the JSON.
    :param species_df: the species DataFrame, as returned from load_species_df().

    :return: the site DataFrame, with the 'species_counts' keys converted from IDs to species names.
    """
    site_df = pd.read_json(path_or_url, orient="index")
    site_df.index.name = "site"
    site_df.columns = ["realm", "ecoregion", "name", "lon", "lat", "num_surveys", "species_counts"]
    site_df["species_counts"] = site_df["species_counts"].map(
        lambda species_counts: {
            species_df.loc[int(species_id)]["name"]: species_count
            for species_id, species_count in species_counts.items()
        }
    )
    return site_df


@st.experimental_memo
def get_selected_area_info(site_df: pd.DataFrame, lat: float, lon: float, radius: float) -> dict:
    """
    Get information about the selected area.

    :param site_df: the site DataFrame, as returned from load_site_df().
    :param lat: the latitude of the selected area.
    :param lon: the longitude of the selected area.
    :param radius: the radius of the selected area in kilometers.

    :return: a dictionary with the following keys and values:
      * 'filtered_site_df': `site_df`, filtered to only include sites within `radius` kilometers from `lat` & `lon`.
      * 'num_surveys': the total number of surveys in the selected area.
      * 'species_freqs': a dictionary mapping species names to their frequencies in the selected area.
    """
    site_distances = site_df.apply(lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1)
    area_site_df = site_df.loc[site_distances <= radius]
    num_area_surveys = area_site_df["num_surveys"].sum()
    area_species_freqs = defaultdict(float)
    for area_site_id, area_site_info in area_site_df.iterrows():
        for species_name, species_count in area_site_info["species_counts"].items():
            area_species_freqs[species_name] += species_count / num_area_surveys
    return dict(filtered_site_df=area_site_df, num_surveys=num_area_surveys, species_freqs=area_species_freqs)
