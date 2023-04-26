"""Utilities to use as part of the streamlit app."""

import typing
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from fastai.learner import Learner, load_learner
from geopy.distance import geodesic

from .constants import DEFAULT_RESOURCES_PATH


def _load_species_df(path_or_url: str | Path) -> pd.DataFrame:
    """
    Load the species DataFrame from the JSON used for other RLS tools.

    :param path_or_url: local path or URL of the JSON.

    :return: the species DataFrame.
    """
    species_df = pd.read_json(path_or_url, orient="index")
    species_df.columns = ["name", "common_names", "url", "method", "images"]
    species_df["method"] = species_df["method"].map({0: "M1", 1: "M2", 2: "Both"})
    species_df["common_name"] = species_df["common_names"].str.split(",", n=1).str[0]
    species_df.drop(columns=["common_names"], inplace=True)
    return species_df


def _load_site_df(path_or_url: str | Path, species_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the site DataFrame from the JSON used for other RLS tools.

    :param path_or_url: local path or URL of the JSON.
    :param species_df: the species DataFrame, as returned from load_species_df().

    :return: the site DataFrame, with the 'species_counts' keys converted from IDs to
             species names.
    """
    site_df = pd.read_json(path_or_url, orient="index")
    site_df.index.name = "site"
    site_df.columns = [
        "realm",
        "ecoregion",
        "name",
        "lon",
        "lat",
        "num_surveys",
        "species_counts",
    ]
    site_df["species_counts"] = site_df["species_counts"].map(
        lambda species_counts: {
            species_df.loc[int(species_id)]["name"]: species_count
            for species_id, species_count in species_counts.items()
        }
    )
    return site_df


@st.cache_data(max_entries=5)
def get_selected_area_info(
    site_df: pd.DataFrame, lat: float, lon: float, radius: float
) -> dict[str, typing.Any]:
    """
    Get information about the selected area.

    :param site_df: the site DataFrame, as returned from load_site_df().
    :param lat: the latitude of the selected area.
    :param lon: the longitude of the selected area.
    :param radius: the radius of the selected area in kilometers.

    :return: a dictionary with the following keys and values:
      * 'filtered_site_df': `site_df`, filtered to only include sites within `radius`
        kilometers from `lat` & `lon`.
      * 'num_surveys': the total number of surveys in the selected area.
      * 'species_freqs': a dictionary mapping species names to their frequencies in the
        selected area.
    """
    site_distances = site_df.apply(
        lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1
    )
    area_site_df = site_df.loc[site_distances <= radius]
    num_area_surveys = area_site_df["num_surveys"].sum()
    area_species_freqs: dict[str, float] = defaultdict(float)
    for _area_site_id, area_site_info in area_site_df.iterrows():
        for species_name, species_count in area_site_info["species_counts"].items():
            area_species_freqs[species_name] += species_count / num_area_surveys
    return dict(
        filtered_site_df=area_site_df,
        num_surveys=num_area_surveys,
        species_freqs=area_species_freqs,
    )


@st.cache_resource
def load_resources(
    resources_path: Path = DEFAULT_RESOURCES_PATH, local_jsons: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, Learner]:
    """
    Load and cache all the static resources used by the streamlit app.

    :param resources_path: path of the resource directory.
    :param local_jsons: if True, append `/local` when loading the survey & species
                        DataFrames.

    :return: a tuple of three items: the species DataFrame, the site DataFrame, and the
             prediction model.
    """
    model = load_learner(resources_path / "model.pkl")
    if local_jsons:
        resources_path /= "local"
    species_df = _load_species_df(resources_path / "api-species.json")
    site_df = _load_site_df(resources_path / "api-site-surveys.json", species_df)
    return species_df, site_df, model
