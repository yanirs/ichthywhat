"""Streamlit fish ID app."""
import sys

from fastai.vision.core import PILImage
import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd

from app_util import get_selected_area_info, load_resources

########################
# Load data and models #
########################
species_df, site_df, model = load_resources(local_species=len(sys.argv) > 1 and sys.argv[1] == "local_species")

########################################
# Show instructions and collect inputs #
########################################

st.title("_Ichthy-what?_ Fishy photo ID :fish:")

st.caption("**Required**: Upload fishy photos to label (cephalopods and other swimmers may work too).")
uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True)

st.caption(
    "**Optional**: Specify where the photos were taken to get occurrence frequencies. You can choose an "
    "existing RLS site or enter coordinates manually."
)
selected_site = st.selectbox(
    "RLS site", ["None: Enter coordinates manually"] + site_df.index.str.cat(site_df.name, ": ").tolist()
)
selected_site_id = selected_site.split(":")[0]
selected_site_info = site_df.loc[selected_site_id] if selected_site_id in site_df.index else pd.DataFrame()

location_columns = st.columns(3)
with location_columns[0]:
    selected_lat = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, step=1.0, value=selected_site_info.get("lat", 0.0)
    )
with location_columns[1]:
    selected_lon = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, step=1.0, value=selected_site_info.get("lon", 0.0)
    )
with location_columns[2]:
    selected_area_radius = st.number_input("Area radius (km)", min_value=0.0, step=10.0, value=500.0)
show_only_area_species = st.checkbox(
    "Only show species known from the area", help="Recommended only if the number of surveys in the area is high"
)

############################################
# Show information about the selected area #
############################################

# Only select an area if the coordinates changed from the 0/0 defaults. This means that this point can't be selected,
# but it's unlikely to be used in practice.
if selected_lat or selected_lon:
    selected_area_info = get_selected_area_info(site_df, selected_lat, selected_lon, selected_area_radius)
else:
    selected_area_info = {}

with st.expander("Location details"):
    st.caption(f"* Past site surveys: {'N/A' if selected_site_info.empty else selected_site_info['num_surveys']}")
    if selected_area_info:
        st.caption(f"* Past area surveys: {selected_area_info['num_surveys']}")
        st.caption(f"* Sites in the area: {len(selected_area_info['filtered_site_df'])}")
        st.caption(f"* Species observed in the area: {len(selected_area_info['species_freqs'])}")
        st.map(selected_area_info["filtered_site_df"], zoom=3)
    else:
        st.caption(f"* Past area surveys: N/A")
        st.caption(f"* Sites in the area: N/A")
        st.caption(f"* Species observed in the area: N/A")

###########################################
# Display results for the uploaded images #
###########################################

for file_index, uploaded_file in enumerate(uploaded_files):
    st.markdown("---")
    st.subheader(f"Uploaded image #{file_index + 1}")
    st.caption(f"Filename: `{uploaded_file.name}`")
    cropped_img = st_cropper(PILImage.create(uploaded_file), box_color="red")

    st.subheader(f"Cropped image for labelling")
    with st.columns(3)[1]:
        st.image(cropped_img)
    st.caption("**Tip:** You're likely to get better matches if you crop the image to show a single fish.")

    num_predictions = 10
    st.subheader(f"Top {num_predictions} matches")

    score_explanation = [
        "**Score explanation:**",
        "* **image**: probability of the uploaded image matching the species according to a model that was trained "
        "only on RLS photos (i.e., without considering location)",
    ]
    if selected_site_info.empty:
        score_explanation.append(
            "* **site**: percentage of the past site surveys where the species was recorded "
            "(N/A, as no site was chosen)"
        )
    else:
        score_explanation.append(
            f"* **site**: percentage of past {selected_site_info['num_surveys']} surveys at "
            f"_{selected_site}_ where the species was recorded"
        )
    if selected_area_info:
        score_explanation.append(
            f"* **area**: percentage of the past {selected_area_info['num_surveys']} surveys at "
            f"the {selected_area_radius}km radius around the given coordinates where the species was "
            f"recorded"
        )
    else:
        score_explanation.append(
            f"* **area**: percentage of past surveys at the {selected_area_radius}km radius around the "
            f"given coordinates where the species was recorded (N/A, as no coordinates were "
            f"specified)"
        )
    st.caption("\n".join(score_explanation))

    # TODO: use TTA? what's the cost?
    # TODO: cache predictions and support changing the number of returned results
    predictions = pd.Series(model.predict(PILImage(cropped_img))[2], index=model.dls.vocab, name="prediction")
    if show_only_area_species:
        predictions = (
            pd.DataFrame(predictions)
            .join(pd.Series(selected_area_info["species_freqs"], name="area_freq"))
            .dropna()["prediction"]
        )
    for prediction_index, (species_name, probability) in enumerate(
        predictions.sort_values(ascending=False).head(num_predictions).items()
    ):
        species_info = species_df[species_df["name"] == species_name].iloc[0]
        common_name = species_info["common_names"].split(", ")[0]
        if selected_site_info.empty:
            site_freq_str = "N/A"
        else:
            site_freq = selected_site_info["species_counts"].get(species_name, 0) / selected_site_info["num_surveys"]
            site_freq_str = f"{100 * site_freq:.1f}%"
        if selected_area_info:
            area_freq_str = f"{100 * selected_area_info['species_freqs'][species_name]:.1f}%"
        else:
            area_freq_str = "N/A"

        info_columns = st.columns([3, 2])
        with info_columns[0]:
            st.markdown(
                f"{prediction_index + 1}) "
                f'[_{species_name}_{f" &ndash; {common_name}" if common_name else ""}]({species_info["url"]})'
            )
        with info_columns[1]:
            st.info(f"**image:** `{probability:.2f}` | **site:** `{site_freq_str}` | **area:** `{area_freq_str}`")

        image_columns = st.columns(3)
        for image_index, image_path_or_url in enumerate(species_info["images"]):
            with image_columns[image_index % 3]:
                st.image(image_path_or_url)
