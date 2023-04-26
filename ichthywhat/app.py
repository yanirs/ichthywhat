"""Streamlit fish ID app."""
import io
import sys

import pandas as pd
import streamlit as st
from app_util import get_selected_area_info, load_resources
from fastai.vision.core import PILImage
from streamlit_cropper import st_cropper

###################################
# Page config: Must be at the top #
###################################

dev_mode = len(sys.argv) > 1 and sys.argv[1] == "dev"
if dev_mode:
    img_root = sys.argv[2]
about_text = """
    This is an _experimental_ web app for fish identification using underwater photos.
    It uses deep learning to find the species that best match the uploaded photos, out
    of over two thousand species recorded on
    [Reef Life Survey Method 1](https://reeflifesurvey.com/methods/) dives.

    Feedback is very welcome! Please send any comments via the
    [Reef Life Survey contact form](https://reeflifesurvey.com/contact/).
"""
st.set_page_config(
    page_title=("[Dev] " if dev_mode else "") + "Ichthy-what? Fishy photo ID",
    page_icon=":tropical_fish:" if dev_mode else ":fish:",
    menu_items={
        "Get help": None,
        "Report a bug": "https://reeflifesurvey.com/contact/",
        "About": about_text,
    },
)

########################
# Load data and models #
########################

species_df, site_df, model = load_resources(local_jsons=dev_mode)

########################################
# Show instructions and collect inputs #
########################################

# Unfortunately, this hack is needed to get the logo to display nicely.
st.markdown(
    '# <a href="https://www.reeflifesurvey.com" target="_blank" rel="noopener">'
    '<img src="https://reeflifesurvey.com/wp-content/uploads/2019/02/'
    'cropped-site-identity-1-192x192.png" '
    'style="float: right; margin-top: 10px; width: 50px"></a> '
    "_Ichthy-what?_ Fishy photo ID",
    unsafe_allow_html=True,
)
st.info(about_text)

st.markdown("---")
st.caption(
    "**Required**: Upload fishy photos to label "
    "(cephalopods and other swimmers may work too)."
)
uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True)

st.markdown("---")
st.caption(
    "**Recommended**: Specify where the photos were taken to get occurrence "
    "frequencies. You can choose an existing RLS site or enter coordinates manually."
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
selected_site_info = (
    site_df.loc[selected_site_id]
    if selected_site_id in site_df.index
    else pd.DataFrame()
)

location_columns = st.columns(3)
with location_columns[0]:
    selected_lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        step=1.0,
        value=selected_site_info.get("lat", 0.0),
    )
with location_columns[1]:
    selected_lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        step=1.0,
        value=selected_site_info.get("lon", 0.0),
    )
with location_columns[2]:
    selected_area_radius = st.number_input(
        "Area radius (km)", min_value=0.0, step=10.0, value=500.0
    )
show_only_area_species = st.checkbox(
    "Only show species known from the area",
    help="Recommended only if the number of surveys in the area is high",
)

############################################
# Show information about the selected area #
############################################

# Only select an area if the coordinates changed from the 0/0 defaults. This means that
# this point can't be selected, but it's unlikely to be used in practice.
if selected_lat or selected_lon:
    selected_area_info = get_selected_area_info(
        site_df, selected_lat, selected_lon, selected_area_radius
    )
else:
    selected_area_info = {}

with st.expander("Location details"):
    st.caption(
        "* Past site surveys: "
        + ("N/A" if selected_site_info.empty else selected_site_info["num_surveys"])
    )
    if selected_area_info:
        st.caption(f"* Past area surveys: {selected_area_info['num_surveys']}")
        st.caption(
            f"* Sites in the area: {len(selected_area_info['filtered_site_df'])}"
        )
        st.caption(
            "* Species observed in the area: "
            + str(len(selected_area_info["species_freqs"]))
        )
        st.map(selected_area_info["filtered_site_df"], zoom=3)
    else:
        st.caption("* Past area surveys: N/A")
        st.caption("* Sites in the area: N/A")
        st.caption("* Species observed in the area: N/A")

########################
# Tweak result display #
########################

st.markdown("---")
st.caption(
    "**Optional**: Set the number of matches to show and choose whether to display the "
    "navigation sidebar. Note that showing a high number of matches may slow things "
    "down, and that the navigation sidebar can be a bit fiddly."
)
num_matches = int(
    st.number_input(
        "Maximum number of matches", min_value=1, max_value=100, step=5, value=10
    )
)
show_navigation_sidebar = st.checkbox("Show navigation sidebar", value=dev_mode)

###########################################
# Display results for the uploaded images #
###########################################

if show_navigation_sidebar and uploaded_files:
    st.sidebar.markdown("[:small_red_triangle: Top](#ichthy-what-fishy-photo-id)")

# Iterate over the uploaded files and display the results.
for file_index, uploaded_file in enumerate(uploaded_files or ()):
    st.markdown("---")

    st.subheader(f":camera: Uploaded image #{file_index + 1}")
    st.caption(f"Filename: `{uploaded_file.name}`")

    st.subheader(":scissors: Cropped image for labelling")
    cropped_img = st_cropper(PILImage.create(uploaded_file), box_color="red")
    cropped_img_columns = st.columns(3)
    if dev_mode:
        # Prepare the cropped image for download in dev mode, where there's a download
        # button for each ID.
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
            decoded_dl_img = PILImage.create(test_dl.decode(test_batch)[0].squeeze())
            st.markdown(f"Decoded: `{decoded_dl_img.size}`")
            st.image(decoded_dl_img)
    else:
        with cropped_img_columns[1]:
            st.image(cropped_img)
    st.caption(
        "**Tip:** You're likely to get better matches if you crop the image to show a "
        "single fish."
    )

    st.subheader(
        f":dizzy: Top {num_matches} {'matches' if num_matches > 1 else 'match'}"
    )

    score_explanation = [
        "**Score explanation:**",
        "* **image**: probability of the uploaded image matching the species according "
        "to a model that was trained only on RLS photos (i.e., without considering "
        "location)",
    ]
    if selected_site_info.empty:
        score_explanation.append(
            "* **site**: percentage of the past site surveys where the species was "
            "recorded (N/A, as no site was chosen)"
        )
    else:
        score_explanation.append(
            f"* **site**: percentage of past {selected_site_info['num_surveys']} "
            f"surveys at _{selected_site}_ where the species was recorded"
        )
    if selected_area_info:
        score_explanation.append(
            f"* **area**: percentage of the past {selected_area_info['num_surveys']} "
            f"surveys at the {selected_area_radius}km radius around the given "
            f"coordinates where the species was recorded"
        )
    else:
        score_explanation.append(
            f"* **area**: percentage of past surveys at the {selected_area_radius}km "
            f"radius around the given coordinates where the species was recorded (N/A, "
            f"as no coordinates were specified)"
        )
    st.caption("\n".join(score_explanation))

    name_filter = st.text_input(
        "Optional: Filter matches by specifying a part of the species name",
        placeholder="Try Leatherjacket, Scarus, or any other relevant word or part of "
        "a word",
        key=f"name-filter-{file_index + 1}",
    ).strip()

    matches = pd.DataFrame(
        dict(probability=model.predict(PILImage(cropped_img))[2], name=model.dls.vocab)
    )
    matches = matches.merge(species_df, on="name")
    if name_filter:
        matches = matches[
            matches["name"].str.contains(name_filter, case=False)
            | matches["common_name"].str.contains(name_filter, case=False)
        ]
    if show_only_area_species and selected_area_info:
        matches = matches.merge(
            pd.Series(selected_area_info["species_freqs"], name="area_freq"),
            left_on="name",
            right_index=True,
        )

    matches = (
        matches.sort_values("probability", ascending=False)
        .head(num_matches)
        .reset_index(drop=True)
    )
    for match in matches.itertuples():
        if selected_site_info.empty:
            site_freq_str = "N/A"
        else:
            site_freq = (
                selected_site_info["species_counts"].get(match.name, 0)
                / selected_site_info["num_surveys"]
            )
            site_freq_str = f"{100 * site_freq:.1f}%"
        if selected_area_info:
            area_freq_str = (
                f"{100 * selected_area_info['species_freqs'][match.name]:.1f}%"
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
                f"**image:** `{match.probability:.2f}` | **site:** `{site_freq_str}` | "
                f"**area:** `{area_freq_str}`"
            )

        image_columns = st.columns(3)
        for image_index, image_path_or_url in enumerate(match.images):
            with image_columns[image_index % 3]:
                st.image(
                    img_root + image_path_or_url if dev_mode else image_path_or_url
                )

        # Only show a sidebar entry for the top match.
        if show_navigation_sidebar and not match.Index:
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                f"[Image #{file_index + 1}: `{uploaded_file.name}`]"
                f"(#uploaded-image-{file_index + 1})"
            )
            st.sidebar.image(cropped_img)
            st.sidebar.markdown(f"**Top match**: {match_link}")

        if dev_mode:
            with st.columns(3)[1]:
                st.download_button(
                    "💾 Save cropped as this ID",
                    cropped_img_bytes,
                    file_name=f"{uploaded_filename_id}C - {match.name}.jpg",
                )
