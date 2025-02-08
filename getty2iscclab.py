"""
MARC LAB Color Visualizer & Restoration Simulator
====================================================

This Streamlit application is designed for cultural heritage conservation professionals.
It processes MARC records to extract color terms, matches these terms against a reference CSV
database of LAB values, visualizes the colors in both 2D and 3D, and provides interactive
tools for simulating color restoration. In addition, the application generates a linked‐data RDF
graph (using rdflib) that relates the MARC record to its associated color terms and detailed LAB
information.

Usage:
    1. Prepare a MARC (.mrc) file containing color terms in subfield 'b'.
    2. Prepare a CSV file with the following columns (case–sensitive):
         - Color Name
         - L   (Lightness, expected range: 0–100)
         - A   (Green–Red chromaticity, typically within ~[-128, 127])
         - B   (Blue–Yellow chromaticity, typically within ~[-128, 127])
    3. Upload both files using the Streamlit interface.
    4. Interact with the visualizations, restoration simulation sliders, and optionally adjust LAB
       values manually.
    5. Download the generated RDF graph (in Turtle format) which links the MARC record to the
       extracted color information, complete with LAB values, hexadecimal code, ISCC–NBS category,
       and (placeholder) Munsell notation.
       
Conservation Science Notes:
    • The use of the CIEDE2000 (ΔE₀₀) color-difference formula is preferred over CIE76 because it
      accounts for the non–uniformity of the LAB space and is more perceptually accurate. This is
      critical in conservation where small color differences may signal important material variations.
    • ISCC–NBS categories, while a simplified color naming system, can provide a useful starting point
      for describing historic colors; however, limitations exist due to its coarse resolution.
    • The restoration simulation tool helps conservation professionals estimate whether color
      adjustments will yield perceptually acceptable matches (e.g., ΔE below ~2 is generally
      imperceptible under controlled conditions).

Dependencies:
    - streamlit, pandas, numpy, plotly, skimage, pymarc, rdflib
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from skimage import color
from dataclasses import dataclass
from typing import Optional, Tuple, List
import io
from pymarc import MARCReader  # Ensure pymarc is installed (pip install pymarc)
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import DCTERMS, SKOS, RDF, RDFS
import uuid
import datetime

# ------------------------------------------------------------------------------
# Constants and Lookup Tables
# ------------------------------------------------------------------------------

# ISCC-NBS Level 2 sRGB centroids (also used for Level 1) for a basic mapping.
ISCC_NBS_CENTROIDS = {
    "pink": (230, 134, 151),
    "red": (185, 40, 66),
    "yellowish pink": (234, 154, 144),
    "reddish orange": (215, 71, 42),
    "reddish brown": (122, 44, 38),
    "orange": (220, 125, 52),
    "brown": (127, 72, 41),
    "orange yellow": (227, 160, 69),
    "yellowish brown": (151, 107, 57),
    "yellow": (217, 180, 81),
    "olive brown": (127, 97, 41),
    "greenish yellow": (208, 196, 69),
    "olive": (114, 103, 44),
    "yellow green": (160, 194, 69),
    "olive green": (62, 80, 31),
    "purplish pink": (229, 137, 191),
    "green": (79, 191, 154),
    "bluish green": (67, 189, 184),
    "greenish blue": (62, 166, 198),
    "blue": (59, 116, 192),
    "purplish blue": (79, 71, 198),
    "violet": (120, 66, 197),
    "purple": (172, 74, 195),
    "reddish purple": (187, 48, 164),
    "white": (231, 225, 233),
    "gray": (147, 142, 147),
    "black": (43, 41, 43)
}

# Custom namespace for LAB values and additional properties.
LABNS = Namespace("http://example.org/lab#")
EX = Namespace("http://example.org/")

# ------------------------------------------------------------------------------
# Data Classes and Color Conversion Functions
# ------------------------------------------------------------------------------

@dataclass
class LABColor:
    """
    Represents a color in the CIELAB color space.
    
    Attributes:
        name (str): The descriptive name of the color.
        L (float): Lightness component (expected 0–100).
        a (float): Chromaticity component (green–red axis).
        b (float): Chromaticity component (blue–yellow axis).
    
    Methods:
        to_rgb: Converts the LAB color to an sRGB tuple.
        to_hex: Converts the LAB color to a hexadecimal string.
    """
    name: str
    L: float
    a: float
    b: float

    def to_rgb(self) -> Tuple[int, int, int]:
        """
        Convert LAB values to sRGB using the CIELAB color space.

        Utilizes skimage.color.lab2rgb which performs gamma correction.
        The resulting values are scaled to [0, 255] and clipped to ensure validity.
        
        Returns:
            Tuple[int, int, int]: The sRGB color as a tuple.
        """
        lab_array = np.array([[[self.L, self.a, self.b]]])
        rgb_array = color.lab2rgb(lab_array)
        rgb = np.clip(rgb_array[0, 0] * 255, 0, 255).astype(int)
        return tuple(rgb.tolist())

    def to_hex(self) -> str:
        """
        Convert the LAB color to a hexadecimal color code.
        
        Returns:
            str: The hex color string.
        """
        rgb = self.to_rgb()
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@dataclass
class MARCRecord:
    """
    Represents a simplified MARC record containing extracted color terms.
    
    Attributes:
        color_terms (List[str]): List of color terms extracted from the MARC record.
        record_id (str): A unique identifier for the record.
    """
    color_terms: List[str]
    record_id: str = ""  # Will be generated if not provided


# ------------------------------------------------------------------------------
# File Parsing and Data Loading Functions
# ------------------------------------------------------------------------------

def parse_marc_with_pymarc(content: bytes) -> MARCRecord:
    """
    Parse a MARC file to extract color terms using pymarc.
    
    This function reads the MARC file (provided as bytes) and extracts values
    from subfield 'b' (assumed to contain descriptive color information). Only the
    first record in the file is processed.
    
    Parameters:
        content (bytes): The byte content of the MARC file.
    
    Returns:
        MARCRecord: An instance containing the list of extracted color terms and a unique record ID.
    
    Raises:
        ValueError: If the MARC content cannot be read or no color terms are found.
    """
    try:
        reader = MARCReader(content, to_unicode=True, force_utf8=True)
    except Exception as e:
        raise ValueError(f"Error reading MARC file: {e}")
    
    color_terms = []
    record_id = str(uuid.uuid4())
    for record in reader:
        for field in record.get_fields():
            if 'b' in field:
                b_values = field.get_subfields('b')
                for value in b_values:
                    term = value.strip().lower()
                    if term:
                        color_terms.append(term)
        break  # Process only the first record

    if not color_terms:
        raise ValueError("No color terms found in the MARC file.")
    return MARCRecord(color_terms=color_terms, record_id=record_id)


@st.cache_data(show_spinner=False)
def cached_load_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    Load and validate CSV data containing color reference information.
    
    This function reads a CSV (provided as bytes) into a DataFrame, validates that the
    required columns exist, converts LAB columns to numeric, and checks that the values
    fall within expected ranges (L: 0–100, a and b roughly within -128 to 127).
    
    Parameters:
        csv_bytes (bytes): The CSV file content as bytes.
    
    Returns:
        pd.DataFrame: A validated DataFrame with columns 'Color Name', 'L', 'A', and 'B'.
    
    Raises:
        ValueError: If required columns are missing or LAB values are invalid.
    """
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    required_columns = {'Color Name', 'L', 'A', 'B'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain the following columns: {required_columns}")
    
    df['Color Name'] = df['Color Name'].str.lower().str.strip()
    for col in ['L', 'A', 'B']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['L', 'A', 'B']].isnull().any().any():
        raise ValueError("One or more LAB value columns contain non-numeric data.")
    
    # Validate value ranges (L: 0–100; A and B: roughly -128 to 127)
    if not df['L'].between(0, 100).all():
        raise ValueError("Some L values are outside the range 0 to 100.")
    if not df['A'].between(-128, 127).all():
        raise ValueError("Some A values are outside the expected range (-128 to 127).")
    if not df['B'].between(-128, 127).all():
        raise ValueError("Some B values are outside the expected range (-128 to 127).")
    
    return df


class ColorDataManager:
    """
    Manages the color reference data loaded from a CSV file.
    
    The CSV must contain the following columns:
      - 'Color Name'
      - 'L' (Lightness)
      - 'A' (Green–Red chromaticity)
      - 'B' (Blue–Yellow chromaticity)
    
    Methods:
        find_color: Retrieves an LABColor for a given color name.
    """
    def __init__(self, csv_bytes: bytes):
        self.df = cached_load_csv(csv_bytes)
    
    def find_color(self, color_name: str) -> Optional[LABColor]:
        """
        Retrieve LABColor information for the specified color name.
        
        Parameters:
            color_name (str): The color term to search for.
            
        Returns:
            Optional[LABColor]: An LABColor instance if found; otherwise, None.
        """
        matched_row = self.df[self.df['Color Name'] == color_name.lower()]
        if not matched_row.empty:
            return LABColor(
                name=color_name,
                L=float(matched_row.iloc[0]['L']),
                a=float(matched_row.iloc[0]['A']),
                b=float(matched_row.iloc[0]['B'])
            )
        return None

# ------------------------------------------------------------------------------
# ISCC–NBS and Additional Color Analysis Functions
# ------------------------------------------------------------------------------

def find_nearest_iscc_category(color: LABColor) -> Tuple[str, float]:
    """
    Determine the nearest ISCC–NBS color category based on sRGB Euclidean distance.
    
    The LABColor is first converted to sRGB, and the Euclidean distance in the sRGB cube is
    computed against predefined ISCC–NBS centroids.
    
    Parameters:
        color (LABColor): The color to categorize.
    
    Returns:
        Tuple[str, float]: The nearest ISCC–NBS category name and the computed distance.
    """
    rgb = np.array(color.to_rgb(), dtype=float)
    min_dist = float('inf')
    nearest_category = None
    for cat_name, cat_rgb in ISCC_NBS_CENTROIDS.items():
        dist = np.linalg.norm(rgb - np.array(cat_rgb, dtype=float))
        if dist < min_dist:
            min_dist = dist
            nearest_category = cat_name
    return nearest_category, min_dist


def compute_delta_e(lab1: LABColor, lab2: LABColor) -> float:
    """
    Compute the color difference (ΔE) between two LAB colors using the CIE76 formula.
    
    CIE76 is simple but does not account for perceptual non–uniformities.
    
    Parameters:
        lab1 (LABColor): First color.
        lab2 (LABColor): Second color.
    
    Returns:
        float: The CIE76 color difference.
    """
    return np.sqrt((lab1.L - lab2.L) ** 2 +
                   (lab1.a - lab2.a) ** 2 +
                   (lab1.b - lab2.b) ** 2)


def compute_delta_e_ciede2000(lab1: LABColor, lab2: LABColor) -> float:
    """
    Compute the color difference (ΔE) using the CIEDE2000 formula.
    
    CIEDE2000 offers improved perceptual accuracy over CIE76, an essential factor in
    conservation where subtle color differences can have significant implications.
    
    Parameters:
        lab1 (LABColor): First color.
        lab2 (LABColor): Second color.
    
    Returns:
        float: The CIEDE2000 color difference.
    """
    L1, a1, b1 = lab1.L, lab1.a, lab1.b
    L2, a2, b2 = lab2.L, lab2.a, lab2.b

    # Mean Lightness
    avg_L = (L1 + L2) / 2.0

    # Compute chroma for each color
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0

    # Adjust a* values to reduce chroma influence on hue difference
    G = 0.5 * (1 - np.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0

    # Hue angles (in degrees)
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        dh = h2_prime - h1_prime
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        delta_h_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh) / 2)

    if C1_prime * C2_prime == 0:
        avg_h_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            avg_h_prime = (h1_prime + h2_prime) / 2
        elif abs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) < 360:
            avg_h_prime = (h1_prime + h2_prime + 360) / 2
        else:
            avg_h_prime = (h1_prime + h2_prime - 360) / 2

    T = (1 - 0.17 * np.cos(np.radians(avg_h_prime - 30)) +
         0.24 * np.cos(np.radians(2 * avg_h_prime)) +
         0.32 * np.cos(np.radians(3 * avg_h_prime + 6)) -
         0.20 * np.cos(np.radians(4 * avg_h_prime - 63)))
    delta_theta = 30 * np.exp(-((avg_h_prime - 275) / 25) ** 2)
    R_C = 2 * np.sqrt((avg_C_prime ** 7) / (avg_C_prime ** 7 + 25 ** 7))
    S_L = 1 + ((0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2))
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    R_T = -np.sin(2 * np.radians(delta_theta)) * R_C

    delta_E = np.sqrt(
        (delta_L_prime / S_L) ** 2 +
        (delta_C_prime / S_C) ** 2 +
        (delta_h_prime / S_H) ** 2 +
        R_T * (delta_C_prime / S_C) * (delta_h_prime / S_H)
    )
    return delta_E


def compute_munsell_notation(color: LABColor) -> str:
    """
    Compute the closest Munsell Notation for the given LAB color.
    
    NOTE: This is a placeholder implementation. In a production environment, you would
    integrate with a dedicated color science library or use an established conversion
    algorithm to derive the closest Munsell color.
    
    Parameters:
        color (LABColor): The color to analyze.
    
    Returns:
        str: A string representing the closest Munsell Notation.
    """
    # Placeholder: Return a dummy Munsell notation
    return "5R 4/14"


def predict_color_deterioration(delta_e: float) -> str:
    """
    Predict potential color deterioration based on the ΔE value.
    
    In conservation, a ΔE below ~2 is generally imperceptible while higher values may indicate
    significant color differences that could reflect material deterioration.
    
    Parameters:
        delta_e (float): The computed color difference.
    
    Returns:
        str: A message indicating the predicted stability or risk.
    """
    if delta_e < 2:
        return "Color appears stable."
    elif delta_e < 5:
        return "Minor deterioration possible."
    else:
        return "Potential significant deterioration."


# ------------------------------------------------------------------------------
# Visualization and UI Utility Functions
# ------------------------------------------------------------------------------

def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app to enhance the visual layout.
    """
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header-container {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #eee;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def display_color_info(color: LABColor):
    """
    Render detailed color information along with a visual swatch.
    
    Displays the color's LAB values, sRGB representation, hexadecimal code, the nearest
    ISCC–NBS category (with computed sRGB distance), and the (placeholder) Munsell notation.
    
    Parameters:
        color (LABColor): The color to display.
    """
    rgb = color.to_rgb()
    hex_color = color.to_hex()
    iscc_category, dist = find_nearest_iscc_category(color)
    munsell = compute_munsell_notation(color)
    color_html = f"""
    <html>
    <head>
    <style>
        .result-container {{
            background-color: #f7f9fc;
            border: 1px solid #cdd9e5;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-family: Arial, sans-serif;
        }}
        .color-info {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 20px;
            align-items: start;
        }}
        .lab-values {{
            background-color: #fff;
            padding: 15px;
            border-radius: 6px;
            font-size: 1.1em;
            line-height: 1.6;
        }}
        .swatch-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }}
        .swatch {{
            width: 200px;
            height: 200px;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            background-color: {hex_color};
        }}
        .swatch-label {{
            font-weight: 500;
            text-align: center;
            color: #333;
        }}
    </style>
    </head>
    <body>
        <div class="result-container">
            <div class="color-info">
                <div class="lab-values">
                    <div><strong>Color Term:</strong> {color.name.title()}</div>
                    <div style="margin-top: 15px;"><strong>LAB Values:</strong><br/>L: {color.L:.2f}<br/>a: {color.a:.2f}<br/>b: {color.b:.2f}</div>
                    <div style="margin-top: 15px;"><strong>sRGB:</strong><br/>RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}<br/>Hex: {hex_color}</div>
                    <div style="margin-top: 15px;"><strong>ISCC–NBS:</strong> {iscc_category.title()}<br/><small>(Distance: {dist:.1f})</small></div>
                    <div style="margin-top: 15px;"><strong>Munsell Notation:</strong> {munsell}</div>
                </div>
                <div class="swatch-container">
                    <div class="swatch"></div>
                    <div class="swatch-label">Color Swatch</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    components.html(color_html, height=400, scrolling=False)


def plot_lab_3d(original: LABColor, adjusted: LABColor):
    """
    Create a 3D scatter plot to visualize the LAB coordinates of the original and adjusted colors.
    
    This visualization helps conservation professionals understand the spatial relationships in
    the perceptually uniform CIELAB space, emphasizing how restoration adjustments shift the color.
    
    Parameters:
        original (LABColor): The original color.
        adjusted (LABColor): The restoration target color.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[original.L],
        y=[original.a],
        z=[original.b],
        mode='markers',
        marker=dict(size=8, color=original.to_hex()),
        name=f"Original: {original.name.title()}"
    ))
    fig.add_trace(go.Scatter3d(
        x=[adjusted.L],
        y=[adjusted.a],
        z=[adjusted.b],
        mode='markers',
        marker=dict(size=8, color=adjusted.to_hex()),
        name="Restoration Target"
    ))
    # Connecting line illustrating ΔE difference
    fig.add_trace(go.Scatter3d(
        x=[original.L, adjusted.L],
        y=[original.a, adjusted.a],
        z=[original.b, adjusted.b],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name="ΔE Difference"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="L (Lightness)",
            yaxis_title="a (Green–Red)",
            zaxis_title="b (Blue–Yellow)"
        ),
        title="CIELAB Color Space Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)


def restoration_tools(matched_colors: List[LABColor]):
    """
    Provide interactive restoration simulation tools for conservation professionals.
    
    Users can adjust LAB values for a restoration target color (via sliders or manual text input)
    and compare it with the original discovered color. Both CIE76 and CIEDE2000 differences are
    computed, and a color deterioration prediction is provided based on the ΔE value.
    
    Parameters:
        matched_colors (List[LABColor]): A list of colors matched from the MARC record.
    """
    st.markdown("## Restoration Simulation Tools")
    st.markdown(
        """
        Adjust the LAB sliders below or enter custom LAB values (as comma-separated values)
        to fine–tune your restoration target color. The computed color differences (ΔE)
        indicate how perceptually similar your adjusted color is to the original. In conservation,
        a ΔE below 2 is typically imperceptible.
        """
    )
    # Select a color from matched colors
    color_names = [color.name.title() for color in matched_colors]
    selected_name = st.selectbox("Select a color for simulation", color_names)
    original_color = next(c for c in matched_colors if c.name.title() == selected_name)

    st.markdown("### Adjust Restoration Target LAB Values")
    # Option for manual LAB input
    custom_lab_input = st.text_input("Or enter custom LAB values (L, a, b)", value="", 
                                     help="Enter three comma-separated numbers (e.g., 50, 0, 0)")
    if custom_lab_input:
        try:
            values = [float(v.strip()) for v in custom_lab_input.split(",")]
            if len(values) != 3:
                st.error("Please enter exactly three values for L, a, and b.")
                return
            target_L, target_a, target_b = values
        except Exception as e:
            st.error(f"Error parsing LAB values: {e}")
            return
    else:
        target_L = st.slider("Target L (Lightness)", min_value=0.0, max_value=100.0,
                             value=float(original_color.L), step=0.1,
                             help="L value: 0 (black) to 100 (white)")
        target_a = st.slider("Target a (Green–Red Axis)", min_value=-128.0, max_value=127.0,
                             value=float(original_color.a), step=0.1,
                             help="Negative values indicate green; positive indicate red.")
        target_b = st.slider("Target b (Blue–Yellow Axis)", min_value=-128.0, max_value=127.0,
                             value=float(original_color.b), step=0.1,
                             help="Negative values indicate blue; positive indicate yellow.")

    adjusted_color = LABColor(name="Restoration Target", L=target_L, a=target_a, b=target_b)

    st.markdown("### Color Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Color**")
        display_color_info(original_color)
    with col2:
        st.markdown("**Restoration Target Color**")
        display_color_info(adjusted_color)

    delta_e_cie76 = compute_delta_e(original_color, adjusted_color)
    delta_e_ciede2000 = compute_delta_e_ciede2000(original_color, adjusted_color)
    deterioration_msg = predict_color_deterioration(delta_e_ciede2000)

    st.markdown(f"### Color Difference (ΔE₇₆ - CIE76): {delta_e_cie76:.2f}")
    st.markdown(f"### Color Difference (ΔE₀₀ - CIEDE2000): {delta_e_ciede2000:.2f}")
    st.markdown(f"### Deterioration Prediction: {deterioration_msg}")
    st.markdown(
        """
        For more information on ΔE thresholds in conservation, see
        [Color Difference and Perception](https://en.wikipedia.org/wiki/Color_difference).
        """
    )
    st.markdown("### 3D LAB Color Space Visualization")
    plot_lab_3d(original_color, adjusted_color)


def generate_rdf_graph(record: MARCRecord, colors: List[LABColor]) -> Graph:
    """
    Generate an RDF graph representing the extracted MARC record and associated color data.
    
    Each MARC record is assigned a unique URI (or uses an existing identifier if present).
    Each color term is given its own URI, and the following triples are created:
      - The MARC record is linked to each color term using dcterms:subject.
      - Each color term is described with its LAB values (using custom predicates),
        hexadecimal color, and ISCC–NBS category.
      - The ISCC–NBS term is included as a skos:narrower property under a broader color category.
    
    Parameters:
        record (MARCRecord): The parsed MARC record.
        colors (List[LABColor]): The list of matched LABColor objects.
    
    Returns:
        Graph: An RDF graph containing the linked color data.
    """
    g = Graph()
    g.bind("dcterms", DCTERMS)
    g.bind("skos", SKOS)
    g.bind("lab", LABNS)
    g.bind("ex", EX)
    
    # Create a unique URI for the MARC record
    record_uri = URIRef(f"http://example.org/marc/{record.record_id}")
    g.add((record_uri, RDF.type, EX.MARCRecord))
    g.add((record_uri, RDFS.label, Literal("MARC Record")))
    g.add((record_uri, DCTERMS.created, Literal(datetime.datetime.now().isoformat())))
    
    for color in colors:
        # Generate a URI for each color term
        color_uri = URIRef(f"http://example.org/marc/{record.record_id}/color/{color.name.replace(' ', '_')}")
        g.add((color_uri, RDF.type, EX.ColorTerm))
        g.add((color_uri, SKOS.prefLabel, Literal(color.name.title())))
        # Add LAB values using custom properties
        g.add((color_uri, LABNS.hasL, Literal(color.L)))
        g.add((color_uri, LABNS.hasA, Literal(color.a)))
        g.add((color_uri, LABNS.hasB, Literal(color.b)))
        g.add((color_uri, LABNS.hasHex, Literal(color.to_hex())))
        # Include the nearest ISCC–NBS category
        iscc_category, _ = find_nearest_iscc_category(color)
        g.add((color_uri, SKOS.narrower, Literal(iscc_category.title())))
        # Add a placeholder for Munsell notation
        g.add((color_uri, LABNS.hasMunsell, Literal(compute_munsell_notation(color))))
        # Link the MARC record to the color term
        g.add((record_uri, DCTERMS.subject, color_uri))
    
    return g


# ------------------------------------------------------------------------------
# Main Application Function
# ------------------------------------------------------------------------------

def main():
    """
    Main function to run the Streamlit application.
    
    This function integrates MARC record parsing, CSV color data lookup, advanced color
    difference calculations (using both CIE76 and CIEDE2000), linked data (RDF) generation,
    and interactive restoration simulation tools to support cultural heritage conservation
    workflows.
    """
    st.set_page_config(page_title="MARC LAB Color Visualizer & Restoration Simulator", layout="wide")
    apply_custom_css()
    
    st.markdown(
        """
        <div class="header-container">
            <h1>MARC LAB Color Visualizer & Restoration Simulator</h1>
            <p>
                Welcome, conservation expert! Upload a MARC (.mrc) file and a CSV with LAB values to visualize 
                discovered colors and assess their historical significance. The app assigns an ISCC–NBS 
                category via sRGB centroid analysis and offers restoration simulation tools for precise color matching.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("📁 File Upload Instructions", expanded=True):
        st.markdown("""
        **MARC File:** Upload a .mrc file containing color terms in subfield 'b'.  
        **CSV File:** Upload a CSV file with columns:  
        - Color Name  
        - L (Lightness, 0–100)  
        - A (Green–Red)  
        - B (Blue–Yellow)
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_marc = st.file_uploader("Upload MARC file (.mrc)", type="mrc", help="Select a MARC file.")
    with col2:
        uploaded_csv = st.file_uploader("Upload LAB values (CSV)", type="csv", help="Select a CSV file with LAB data.")
    
    if uploaded_marc and uploaded_csv:
        try:
            # Parse MARC file to extract color terms
            marc_content = uploaded_marc.read()
            record_data = parse_marc_with_pymarc(marc_content)
            # Load CSV color reference data
            csv_bytes = uploaded_csv.read()
            color_manager = ColorDataManager(csv_bytes)
            matched_colors = []
            for term in record_data.color_terms:
                color_obj = color_manager.find_color(term)
                if color_obj:
                    matched_colors.append(color_obj)
            
            if matched_colors:
                st.markdown("### Matched Colors from MARC File")
                for color_obj in matched_colors:
                    display_color_info(color_obj)
                with st.expander("📋 All Color Terms Found"):
                    st.write(", ".join(record_data.color_terms))
                # Launch restoration simulation tools
                restoration_tools(matched_colors)
                
                # Generate and offer RDF graph download
                rdf_graph = generate_rdf_graph(record_data, matched_colors)
                rdf_data = rdf_graph.serialize(format="turtle")
                st.download_button(label="Download RDF Graph (Turtle)",
                                   data=rdf_data,
                                   file_name="marc_colors.ttl",
                                   mime="text/turtle")
            else:
                st.error("No matching color terms found in the CSV file.")
        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
