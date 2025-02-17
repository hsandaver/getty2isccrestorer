"""
MARC LAB Color Visualizer & Restoration Simulator with Enhanced Fading Simulator and MARC Metadata Enrichment
============================================================================================================

This Streamlit application is designed for cultural heritage conservation professionals.
It processes MARC records to extract color terms and metadata (including the OCLC number),
matches these terms against a reference CSV database of LAB values, visualizes the colors in both 2D and 3D,
provides interactive tools for simulating color restoration and fading, and generates an RDF graph.
The RDF graph now includes additional MARC record metadata (e.g., the OCLC number) to help users identify the record.

Usage:
    1. Prepare a MARC (.mrc) file containing color terms in subfield 'b' and an OCLC number in field "035" (subfield 'a').
    2. Prepare a CSV file with columns (case–sensitive):
         - Color Name
         - L   (Lightness, 0–100)
         - A   (Green–Red)
         - B   (Blue–Yellow)
    3. Upload both files using the Streamlit interface.
    4. Interact with the restoration simulation tools and the enhanced fading simulator.
    5. Download the generated RDF graph (in Turtle format) linking the MARC record to its enriched color data.

Dependencies:
    - streamlit, pandas, numpy, plotly, scikit-image, pymarc, rdflib
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

LABNS = Namespace("http://example.org/lab#")
EX = Namespace("http://example.org/")

# ------------------------------------------------------------------------------
# Data Classes and Color Conversion Functions
# ------------------------------------------------------------------------------

@dataclass
class LABColor:
    name: str
    L: float
    a: float
    b: float

    def to_rgb(self) -> Tuple[int, int, int]:
        lab_array = np.array([[[self.L, self.a, self.b]]])
        rgb_array = color.lab2rgb(lab_array)
        rgb = np.clip(rgb_array[0, 0] * 255, 0, 255).astype(int)
        return tuple(rgb.tolist())

    def to_hex(self) -> str:
        rgb = self.to_rgb()
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@dataclass
class MARCRecord:
    color_terms: List[str]
    record_id: str = ""
    oclc: Optional[str] = None  # Store the OCLC number if available


# ------------------------------------------------------------------------------
# File Parsing and Data Loading Functions
# ------------------------------------------------------------------------------

def parse_marc_with_pymarc(content: bytes) -> MARCRecord:
    try:
        reader = MARCReader(content, to_unicode=True, force_utf8=True)
    except Exception as e:
        raise ValueError(f"Error reading MARC file: {e}")
    
    color_terms = []
    record_id = str(uuid.uuid4())
    oclc = None
    for record in reader:
        # Extract color terms from any field containing subfield 'b'
        for field in record.get_fields():
            if 'b' in field:
                b_values = field.get_subfields('b')
                for value in b_values:
                    term = value.strip().lower()
                    if term:
                        color_terms.append(term)
        # Attempt to extract the OCLC number from field "035", subfield 'a'
        for field in record.get_fields("035"):
            subfields = field.get_subfields('a')
            for sub in subfields:
                if "OCoLC" in sub:
                    oclc = sub.strip()
                    break
            if oclc:
                break
        break  # Process only the first record

    if not color_terms:
        raise ValueError("No color terms found in the MARC file.")
    return MARCRecord(color_terms=color_terms, record_id=record_id, oclc=oclc)


@st.cache_data(show_spinner=False)
def cached_load_csv(csv_bytes: bytes) -> pd.DataFrame:
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
    
    if not df['L'].between(0, 100).all():
        raise ValueError("Some L values are outside the range 0 to 100.")
    if not df['A'].between(-128, 127).all():
        raise ValueError("Some A values are outside the expected range (-128 to 127).")
    if not df['B'].between(-128, 127).all():
        raise ValueError("Some B values are outside the expected range (-128 to 127).")
    
    return df


class ColorDataManager:
    def __init__(self, csv_bytes: bytes):
        self.df = cached_load_csv(csv_bytes)
    
    def find_color(self, color_name: str) -> Optional[LABColor]:
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
    return np.sqrt((lab1.L - lab2.L) ** 2 +
                   (lab1.a - lab2.a) ** 2 +
                   (lab1.b - lab2.b) ** 2)


def compute_delta_e_ciede2000(lab1: LABColor, lab2: LABColor) -> float:
    L1, a1, b1 = lab1.L, lab1.a, lab1.b
    L2, a2, b2 = lab2.L, lab2.a, lab2.b
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0
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
    return "5R 4/14"  # Placeholder implementation


def predict_color_deterioration(delta_e: float) -> str:
    if delta_e < 2:
        return "Color appears stable."
    elif delta_e < 5:
        return "Minor deterioration possible."
    else:
        return "Potential significant deterioration."


# ------------------------------------------------------------------------------
# Enhanced Fading Simulator Functions
# ------------------------------------------------------------------------------

def calculate_delta_e_fading(original: LABColor, light: float, humidity: float, time: float, dye_type: str) -> float:
    """
    Calculate the color difference (ΔE) for fading given environmental parameters.
    Empirical model: ΔE = k * light * humidity * time,
    where k is determined by the dye type.
    """
    if dye_type.lower() == 'natural':
        k = 1e-6  # Example constant: natural dyes fade faster
    elif dye_type.lower() == 'synthetic':
        k = 5e-7  # Example constant: synthetic dyes are more stable
    else:
        raise ValueError("dye_type must be 'natural' or 'synthetic'")
    
    delta_e = k * light * humidity * time
    return delta_e


def time_to_fade(original: LABColor, light: float, humidity: float, dye_type: str, target_delta_e: float = 2.0) -> float:
    """
    Estimate the time (in hours) required for the color to fade such that ΔE reaches target_delta_e.
    Uses the inverted model: time = target_delta_e / (k * light * humidity)
    """
    if dye_type.lower() == 'natural':
        k = 1e-6
    elif dye_type.lower() == 'synthetic':
        k = 5e-7
    else:
        raise ValueError("dye_type must be 'natural' or 'synthetic'")
    
    if light * humidity * k == 0:
        return float('inf')
    
    time_required = target_delta_e / (k * light * humidity)
    return time_required


def format_time(total_hours: float) -> str:
    """
    Convert a time in hours to a string in years, days, hours, and minutes.
    Assumes: 1 year = 365 days, 1 day = 24 hours, 1 hour = 60 minutes.
    """
    total_minutes = int(round(total_hours * 60))
    years = total_minutes // 525600  # 525600 minutes in a year
    rem_minutes = total_minutes % 525600
    days = rem_minutes // 1440       # 1440 minutes in a day
    rem_minutes %= 1440
    hours = rem_minutes // 60
    minutes = rem_minutes % 60

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        return "0 minutes"
    return ", ".join(parts)


# ------------------------------------------------------------------------------
# Visualization and UI Utility Functions
# ------------------------------------------------------------------------------

def apply_custom_css():
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
    st.markdown("## Restoration Simulation Tools")
    st.markdown(
        """
        Adjust the LAB sliders below or enter custom LAB values (as comma-separated values)
        to fine–tune your restoration target color. The computed color differences (ΔE)
        indicate how perceptually similar your adjusted color is to the original. In conservation,
        a ΔE below 2 is typically imperceptible.
        """
    )
    color_names = [color.name.title() for color in matched_colors]
    selected_name = st.selectbox("Select a color for simulation", color_names)
    original_color = next(c for c in matched_colors if c.name.title() == selected_name)

    st.markdown("### Adjust Restoration Target LAB Values")
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


def fading_simulator(matched_colors: List[LABColor]):
    st.markdown("## Enhanced Fading Simulator")
    fade_color_names = [color.name.title() for color in matched_colors]
    selected_fade_name = st.selectbox("Select a color for fading simulation", fade_color_names, key="fading_simulation")
    original_fade_color = next(c for c in matched_colors if c.name.title() == selected_fade_name)
    
    st.markdown("### Environmental Parameters for Fading Simulation")
    light_intensity = st.number_input("Light Intensity (lux)", min_value=0.0, value=5000.0, step=100.0)
    humidity = st.slider("Relative Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    dye_type = st.selectbox("Dye Type", options=["Natural", "Synthetic"])
    
    st.markdown("### Fading Simulation")
    time_exposure = st.slider("Exposure Time (hours)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
    computed_delta_e = calculate_delta_e_fading(original_fade_color, light_intensity, humidity, time_exposure, dye_type)
    st.markdown(f"Computed ΔE after {time_exposure} hours: **{computed_delta_e:.2f}**")
    
    target_time = time_to_fade(original_fade_color, light_intensity, humidity, dye_type)
    formatted_time = format_time(target_time)
    st.markdown(f"Estimated time to reach ΔE of 2: **{formatted_time}**")
    
    retest_note = st.text_input("Note: When to re-test for fading (e.g., 'After 6 months')", key="fading_note")
    if retest_note:
        st.info(f"Re-test note: {retest_note}")


def generate_rdf_graph(record: MARCRecord, colors: List[LABColor]) -> Graph:
    g = Graph()
    g.bind("dcterms", DCTERMS)
    g.bind("skos", SKOS)
    g.bind("lab", LABNS)
    g.bind("ex", EX)
    
    record_uri = URIRef(f"http://example.org/marc/{record.record_id}")
    g.add((record_uri, RDF.type, EX.MARCRecord))
    g.add((record_uri, RDFS.label, Literal("MARC Record")))
    g.add((record_uri, DCTERMS.created, Literal(datetime.datetime.now().isoformat())))
    # Add the OCLC number as an identifier if available
    if record.oclc:
        g.add((record_uri, DCTERMS.identifier, Literal(record.oclc)))
    
    for color in colors:
        color_uri = URIRef(f"http://example.org/marc/{record.record_id}/color/{color.name.replace(' ', '_')}")
        g.add((color_uri, RDF.type, EX.ColorTerm))
        g.add((color_uri, SKOS.prefLabel, Literal(color.name.title())))
        g.add((color_uri, LABNS.hasL, Literal(color.L)))
        g.add((color_uri, LABNS.hasA, Literal(color.a)))
        g.add((color_uri, LABNS.hasB, Literal(color.b)))
        g.add((color_uri, LABNS.hasHex, Literal(color.to_hex())))
        iscc_category, _ = find_nearest_iscc_category(color)
        g.add((color_uri, SKOS.narrower, Literal(iscc_category.title())))
        g.add((color_uri, LABNS.hasMunsell, Literal(compute_munsell_notation(color))))
        g.add((record_uri, DCTERMS.subject, color_uri))
    
    return g


# ------------------------------------------------------------------------------
# Main Application Function
# ------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="MARC LAB Color Visualizer & Restoration Simulator", layout="wide")
    apply_custom_css()
    
    st.markdown(
        """
        <div class="header-container">
            <h1>MARC LAB Color Visualizer & Restoration Simulator</h1>
            <p>
                Welcome, conservation expert! Upload a MARC (.mrc) file and a CSV with LAB values to visualize 
                discovered colors and assess their historical significance. The app assigns an ISCC–NBS 
                category via sRGB centroid analysis, offers restoration simulation tools, and now includes
                an enhanced fading simulator that calculates fading based on light, humidity, and time.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("📁 File Upload Instructions", expanded=True):
        st.markdown("""
        **MARC File:** Upload a .mrc file containing color terms in subfield 'b' and an OCLC number in field "035" (subfield 'a').  
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
            marc_content = uploaded_marc.read()
            record_data = parse_marc_with_pymarc(marc_content)
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
                restoration_tools(matched_colors)
                fading_simulator(matched_colors)
                
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
