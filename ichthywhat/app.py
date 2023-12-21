"""Streamlit fish ID app."""
import dataclasses
import io
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
import streamlit as st
from fastai.learner import Learner, load_learner
from fastai.vision.core import PILImage
from geopy.distance import geodesic
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_cropper import st_cropper

from ichthywhat.constants import DEFAULT_RESOURCES_PATH
from ichthywhat.inference import OnnxWrapper

_ABOUT_TEXT: Final[
    str
] = """
    This is an _experimental_ web app for fish identification using underwater photos.
    It uses deep learning to find the species that best match the uploaded photos, out
    of over two thousand species recorded on
    [Reef Life Survey Method 1](https://reeflifesurvey.com/methods/) dives.

    Feedback is very welcome! Please send any comments via the
    [Reef Life Survey contact form](https://reeflifesurvey.com/contact/).
"""
_EMPTY_DF: Final[pd.DataFrame] = pd.DataFrame()


@dataclasses.dataclass
class AppOptions:
    """Convenience class for passing around options between the app's functions."""

    dev_mode: bool = False
    img_root: str = ""
    uploaded_files: Sequence[UploadedFile] = ()
    selected_site_info: pd.DataFrame = _EMPTY_DF
    selected_lat: float = 0
    selected_lon: float = 0
    selected_area_radius: float = 500
    selected_area_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    show_only_area_species: bool = False
    num_matches: int = 10
    show_navigation_sidebar: bool = False


def main() -> None:
    """Run the streamlit app."""
    opts = AppOptions()
    opts.dev_mode = len(sys.argv) > 1 and sys.argv[1] == "dev"
    if opts.dev_mode:
        opts.img_root = sys.argv[2]
    st.set_page_config(
        page_title=("[Dev] " if opts.dev_mode else "") + "Ichthy-what? Fishy photo ID",
        page_icon=":tropical_fish:" if opts.dev_mode else ":fish:",
        menu_items={
            "Get help": None,
            "Report a bug": "https://reeflifesurvey.com/contact/",
            "About": _ABOUT_TEXT,
        },
    )
    model = _load_model(opts.dev_mode)
    df_path = (
        DEFAULT_RESOURCES_PATH / "local" if opts.dev_mode else DEFAULT_RESOURCES_PATH
    )
    species_df = _load_species_df(df_path / "api-species.json")
    site_df = _load_site_df(df_path / "api-site-surveys.json", species_df)
    _display_basic_inputs(opts, site_df)
    _display_selected_area_info(opts, site_df)
    _display_view_options(opts)
    _display_results(opts, species_df, model)


@st.cache_resource
def _load_model(dev_mode: bool) -> Learner | OnnxWrapper:
    if dev_mode:
        return load_learner(DEFAULT_RESOURCES_PATH / "model.pkl")
    return OnnxWrapper(DEFAULT_RESOURCES_PATH / "model.onnx")


@st.cache_resource
def _load_species_df(path_or_url: str | Path) -> pd.DataFrame:
    """Load the species DataFrame from the JSON used for other RLS tools.

    Parameters
    ----------
    path_or_url
        local path or URL of the JSON.

    Returns
    -------
    pd.DataFrame
        the species DataFrame.
    """
    species_df = pd.read_json(path_or_url, orient="index")
    species_df.columns = ["name", "common_names", "url", "method", "images"]
    species_df["method"] = species_df["method"].map({0: "M1", 1: "M2", 2: "Both"})
    species_df["common_name"] = species_df["common_names"].str.split(",", n=1).str[0]
    species_df.drop(columns=["common_names"], inplace=True)
    return species_df


@st.cache_resource
def _load_site_df(path_or_url: str | Path, species_df: pd.DataFrame) -> pd.DataFrame:
    """Load the site DataFrame from the JSON used for other RLS tools.

    Parameters
    ----------
    path_or_url
        local path or URL of the JSON.
    species_df
        the species DataFrame, as returned from load_species_df().

    Returns
    -------
    pd.DataFrame
        the site DataFrame, with the 'species_counts' keys converted from IDs to species
        names.
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
            species_df.loc[species_id]["name"]: species_count
            for species_id, species_count in species_counts.items()
        }
    )
    return site_df


@st.cache_data(max_entries=5)
def _get_selected_area_info(
    site_df: pd.DataFrame, lat: float, lon: float, radius: float
) -> dict[str, Any]:
    """Get information about the selected area.

    Parameters
    ----------
    site_df
        the site DataFrame, as returned from load_site_df().
    lat
        the latitude of the selected area.
    lon
        the longitude of the selected area.
    radius
        the radius of the selected area in kilometers.

    Returns
    -------
    dict[str, Any]
        a dictionary with the following keys and values:
        * 'filtered_site_df': `site_df`, filtered to only include sites within `radius`
          kilometers from `lat` & `lon`.
        * 'num_surveys': the total number of surveys in the selected area.
        * 'species_freqs': a dictionary mapping species names to their frequencies in
          the selected area.
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


def _display_basic_inputs(opts: AppOptions, site_df: pd.DataFrame) -> None:
    # Unfortunately, this hack is needed to get the logo to display nicely.
    st.markdown(
        '# <a href="https://www.reeflifesurvey.com" target="_blank" rel="noopener">'
        '<img src="https://reeflifesurvey.com/wp-content/uploads/2019/02/'
        'cropped-site-identity-1-192x192.png" '
        'style="float: right; margin-top: 10px; width: 50px"></a> '
        "_Ichthy-what?_ Fishy photo ID",
        unsafe_allow_html=True,
    )
    st.info(_ABOUT_TEXT)

    st.markdown("---")
    st.caption(
        "**Required**: Upload fishy photos to label "
        "(cephalopods and other swimmers may work too)."
    )
    opts.uploaded_files = (
        st.file_uploader("Choose image files", accept_multiple_files=True) or ()
    )

    st.markdown("---")
    st.caption(
        "**Recommended**: Specify where the photos were taken to get occurrence "
        "frequencies. You can choose an existing RLS site or enter coordinates "
        "manually."
    )
    selected_site = (
        st.selectbox(
            "RLS site",
            [
                "None: Enter coordinates manually",
                *site_df.index.str.cat(site_df["name"], ": ").tolist(),
            ],
        )
        or "None:"
    )
    selected_site_id = selected_site.split(":")[0]
    opts.selected_site_info = (
        site_df.loc[selected_site_id]
        if selected_site_id in site_df.index
        else _EMPTY_DF
    )

    location_columns = st.columns(3)
    with location_columns[0]:
        opts.selected_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            step=1.0,
            value=opts.selected_site_info.get("lat", 0.0),
        )
    with location_columns[1]:
        opts.selected_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            step=1.0,
            value=opts.selected_site_info.get("lon", 0.0),
        )
    with location_columns[2]:
        opts.selected_area_radius = st.number_input(
            "Area radius (km)", min_value=0.0, step=10.0, value=500.0
        )
    opts.show_only_area_species = st.checkbox(
        "Only show species known from the area",
        help="Recommended only if the number of surveys in the area is high",
    )


def _display_selected_area_info(opts: AppOptions, site_df: pd.DataFrame) -> None:
    # Only select an area if the coordinates changed from the 0/0 defaults. This means
    # that this point can't be selected, but it's unlikely to be used in practice.
    if opts.selected_lat or opts.selected_lon:
        opts.selected_area_info = _get_selected_area_info(
            site_df, opts.selected_lat, opts.selected_lon, opts.selected_area_radius
        )
    else:
        opts.selected_area_info = {}

    with st.expander("Location details"):
        st.caption(
            f"* Past site surveys: {opts.selected_site_info.get('num_surveys', 'N/A')}"
        )
        if opts.selected_area_info:
            st.caption(f"* Past area surveys: {opts.selected_area_info['num_surveys']}")
            st.caption(
                "* Sites in the area: "
                + str(len(opts.selected_area_info["filtered_site_df"]))
            )
            st.caption(
                "* Species observed in the area: "
                + str(len(opts.selected_area_info["species_freqs"]))
            )
            st.map(opts.selected_area_info["filtered_site_df"], zoom=3)
        else:
            st.caption("* Past area surveys: N/A")
            st.caption("* Sites in the area: N/A")
            st.caption("* Species observed in the area: N/A")


def _display_view_options(opts: AppOptions) -> None:
    st.markdown("---")
    st.caption(
        "**Optional**: Set the number of matches to show and choose whether to display "
        "the navigation sidebar. Note that showing a high number of matches may slow "
        "things down, and that the navigation sidebar can be a bit fiddly."
    )
    opts.num_matches = int(
        st.number_input(
            "Maximum number of matches", min_value=1, max_value=100, step=5, value=10
        )
    )
    opts.show_navigation_sidebar = st.checkbox(
        "Show navigation sidebar", value=opts.dev_mode
    )


def _classify_image(model: Learner | OnnxWrapper, img: Image.Image) -> pd.DataFrame:
    if isinstance(model, OnnxWrapper):
        return pd.DataFrame(
            pd.Series(model.predict(img), name="probability")
        ).reset_index(names="name")
    return pd.DataFrame(
        dict(
            probability=model.predict(PILImage(img))[2],
            name=model.dls.vocab,
        )
    ).sort_values("probability", ascending=False)


def _display_results(
    opts: AppOptions, species_df: pd.DataFrame, model: Learner | OnnxWrapper
) -> None:
    if opts.show_navigation_sidebar and opts.uploaded_files:
        st.sidebar.markdown("[:small_red_triangle: Top](#ichthy-what-fishy-photo-id)")

    # Iterate over the uploaded files and display the results.
    for file_index, uploaded_file in enumerate(opts.uploaded_files):
        st.markdown("---")

        st.subheader(f":camera: Uploaded image #{file_index + 1}")
        st.caption(f"Filename: `{uploaded_file.name}`")

        st.subheader(":scissors: Cropped image for labelling")
        cropped_img = st_cropper(PILImage.create(uploaded_file), box_color="red")
        cropped_img_columns = st.columns(3)
        if opts.dev_mode:
            assert isinstance(model, Learner)
            # Prepare the cropped image for download in dev mode, where there's a
            # download button for each ID.
            with io.BytesIO() as cropped_img_file:
                cropped_img.save(cropped_img_file, format="JPEG")
                cropped_img_bytes = cropped_img_file.getvalue()
            uploaded_filename_id = uploaded_file.name.split(".")[0].split()[0]

            # Show extra debug info on the cropped image in dev mode.
            with cropped_img_columns[0]:
                st.markdown(f"Crop: `{cropped_img.size}`")
                st.image(cropped_img)
            with cropped_img_columns[1]:
                test_dl = model.dls.test_dl([PILImage(cropped_img)], num_workers=0)
                test_batch = test_dl.one_batch()
                undecoded_dl_img = PILImage.create(test_batch[0].squeeze())
                st.markdown(f"Undecoded: `{undecoded_dl_img.size}`")
                st.image(undecoded_dl_img)
            with cropped_img_columns[2]:
                decoded_dl_img = PILImage.create(
                    test_dl.decode(test_batch)[0].squeeze()
                )
                st.markdown(f"Decoded: `{decoded_dl_img.size}`")
                st.image(decoded_dl_img)
        else:
            with cropped_img_columns[1]:
                st.image(cropped_img)
        st.caption(
            "**Tip:** You're likely to get better matches if you crop the image to "
            "show a single fish."
        )

        st.subheader(
            f":dizzy: Top {opts.num_matches} "
            f"{'matches' if opts.num_matches > 1 else 'match'}"
        )

        score_explanation = [
            "**Score explanation:**",
            "* **image**: probability of the uploaded image matching the species "
            "according to a model that was trained only on RLS photos (i.e., without "
            "considering location)",
        ]
        if opts.selected_site_info.empty:
            score_explanation.append(
                "* **site**: percentage of the past site surveys where the species was "
                "recorded (N/A, as no site was chosen)"
            )
        else:
            score_explanation.append(
                "* **site**: percentage of past "
                f"{opts.selected_site_info['num_surveys']} surveys at "
                f"_{opts.selected_site_info['name']}_ where the species was recorded"
            )
        if opts.selected_area_info:
            score_explanation.append(
                f"* **area**: percentage of the past "
                f"{opts.selected_area_info['num_surveys']} surveys at the "
                f"{opts.selected_area_radius}km radius around the given coordinates "
                f"where the species was recorded"
            )
        else:
            score_explanation.append(
                f"* **area**: percentage of past surveys at the "
                f"{opts.selected_area_radius}km radius around the given coordinates "
                "where the species was recorded (N/A, as no coordinates were specified)"
            )
        st.caption("\n".join(score_explanation))

        name_filter = st.text_input(
            "Optional: Filter matches by specifying a part of the species name",
            placeholder="Try Leatherjacket, Scarus, or any other relevant word or part "
            "of a word",
            key=f"name-filter-{file_index + 1}",
        ).strip()

        matches = _classify_image(model, cropped_img).merge(species_df, on="name")
        if name_filter:
            matches = matches[
                matches["name"].str.contains(name_filter, case=False)
                | matches["common_name"].str.contains(name_filter, case=False)
            ]
        if opts.show_only_area_species and opts.selected_area_info:
            matches = matches.merge(
                pd.Series(opts.selected_area_info["species_freqs"], name="area_freq"),
                left_on="name",
                right_index=True,
            )
        matches = matches.head(opts.num_matches).reset_index(drop=True)
        for match in matches.itertuples():
            if opts.selected_site_info.empty:
                site_freq_str = "N/A"
            else:
                site_freq = (
                    opts.selected_site_info["species_counts"].get(match.name, 0)
                    / opts.selected_site_info["num_surveys"]
                )
                site_freq_str = f"{100 * site_freq:.1f}%"
            if opts.selected_area_info:
                area_freq_str = (
                    f"{100 * opts.selected_area_info['species_freqs'][match.name]:.1f}%"
                )
            else:
                area_freq_str = "N/A"

            info_columns = st.columns([3, 2])
            with info_columns[0]:
                match_link = (
                    f"[_{match.name}_"
                    f'{f" &ndash; {match.common_name}" if match.common_name else ""}]'
                    f"({match.url})"
                )
                st.markdown(f"{match.Index + 1}) {match_link}")
            with info_columns[1]:
                st.info(
                    f"**image:** `{match.probability:.2f}` | "
                    f"**site:** `{site_freq_str}` | "
                    f"**area:** `{area_freq_str}`"
                )

            image_columns = st.columns(3)
            for image_index, image_path_or_url in enumerate(match.images):
                with image_columns[image_index % 3]:
                    st.image(
                        image_path_or_url.replace("/img/", opts.img_root)
                        if opts.img_root
                        else image_path_or_url
                    )

            # Only show a sidebar entry for the top match.
            if opts.show_navigation_sidebar and not match.Index:
                st.sidebar.markdown("---")
                st.sidebar.markdown(
                    f"[Image #{file_index + 1}: `{uploaded_file.name}`]"
                    f"(#uploaded-image-{file_index + 1})"
                )
                st.sidebar.image(cropped_img)
                st.sidebar.markdown(f"**Top match**: {match_link}")

            if opts.dev_mode:
                with st.columns(3)[1]:
                    st.download_button(
                        "💾 Save cropped as this ID",
                        cropped_img_bytes,
                        file_name=f"{uploaded_filename_id}C - {match.name}.jpg",
                    )


if __name__ == "__main__":
    main()
