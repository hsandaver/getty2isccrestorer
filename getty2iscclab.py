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

# ------------------------------------------------------------------------------
# Built-in lookup table of ISCC-NBS Level 2 sRGB centroids (also used for Level 1)
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

# ------------------------------------------------------------------------------
# Data classes and conversion functions
# ------------------------------------------------------------------------------

@dataclass
class LABColor:
    """
    Represents a color in the CIELAB color space.
    
    Attributes:
        name: The descriptive name of the color.
        L: Lightness component (0 to 100).
        a: Chromaticity component (green–red axis).
        b: Chromaticity component (blue–yellow axis).
    
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
        Utilizes skimage.color.lab2rgb which includes proper gamma correction.
        Values are scaled to [0, 255] and clipped to ensure validity.
        """
        lab_array = np.array([[[self.L, self.a, self.b]]])
        rgb_array = color.lab2rgb(lab_array)
        rgb = np.clip(rgb_array[0, 0] * 255, 0, 255).astype(int)
        return tuple(rgb.tolist())

    def to_hex(self) -> str:
        """
        Convert LAB values to a hex color code.
        """
        rgb = self.to_rgb()
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@dataclass
class MARCRecord:
    """
    Represents a simplified MARC record containing color terms.
    
    Attributes:
        color_terms: List of color terms extracted from the MARC record.
    """
    color_terms: List[str]


def parse_marc_with_pymarc(content: bytes) -> MARCRecord:
    """
    Parse a MARC file using pymarc and extract color terms.
    
    The function iterates over the MARC record fields, specifically targeting 
    subfields 'b' that are assumed to contain descriptive color information.
    
    Parameters:
        content: Byte content of the MARC file.
    
    Returns:
        A MARCRecord instance containing the extracted color terms.
    
    Note:
        Only the first record in the file is processed.
    """
    reader = MARCReader(content, to_unicode=True, force_utf8=True)
    color_terms = []
    for record in reader:
        for field in record.get_fields():
            if 'b' in field:
                b_values = field.get_subfields('b')
                for value in b_values:
                    color_terms.append(value.strip().lower())
        break  # Only process the first record
    return MARCRecord(color_terms=color_terms)


class ColorDataManager:
    """
    Manages the color reference data loaded from a CSV file.
    
    The CSV file must contain the following columns:
      - 'Color Name'
      - 'L' (Lightness)
      - 'A' (Green–Red chromaticity)
      - 'B' (Blue–Yellow chromaticity)
    
    Methods:
        find_color: Retrieves an LABColor for a given color name.
    """
    def __init__(self, csv_data: io.BytesIO):
        self.df = self._load_csv(csv_data)
    
    @staticmethod
    def _load_csv(csv_data: io.BytesIO) -> pd.DataFrame:
        """
        Load and validate CSV data containing color reference information.
        
        Raises:
            ValueError: If required columns are missing or if non-numeric values are found.
        """
        df = pd.read_csv(csv_data)
        required_columns = {'Color Name', 'L', 'A', 'B'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        df['Color Name'] = df['Color Name'].str.lower().str.strip()
        # Ensure the LAB columns contain numeric values
        for col in ['L', 'A', 'B']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[['L', 'A', 'B']].isnull().any().any():
            raise ValueError("One or more LAB value columns contain non-numeric data.")
        return df
    
    def find_color(self, color_name: str) -> Optional[LABColor]:
        """
        Retrieve LABColor information for the specified color name.
        
        Parameters:
            color_name: The color term to search for.
            
        Returns:
            An LABColor instance if found; otherwise, None.
        """
        color_row = self.df[self.df['Color Name'] == color_name.lower()]
        if not color_row.empty:
            return LABColor(
                name=color_name,
                L=float(color_row.iloc[0]['L']),
                a=float(color_row.iloc[0]['A']),
                b=float(color_row.iloc[0]['B'])
            )
        return None

# ------------------------------------------------------------------------------
# ISCC–NBS category lookup using sRGB centroid table
# ------------------------------------------------------------------------------

def find_nearest_iscc_category(color: LABColor) -> Tuple[str, float]:
    """
    Determine the nearest ISCC-NBS color category based on sRGB distance.
    
    The LABColor is first converted to sRGB, and the Euclidean distance in the 
    sRGB color cube is computed against predefined ISCC-NBS centroids.
    
    Parameters:
        color: The LABColor instance to categorize.
    
    Returns:
        A tuple containing the nearest category name and the computed distance.
    """
    rgb = np.array(color.to_rgb(), dtype=float)
    min_dist = float('inf')
    nearest_category = None
    for cat, cat_rgb in ISCC_NBS_CENTROIDS.items():
        dist = np.linalg.norm(rgb - np.array(cat_rgb, dtype=float))
        if dist < min_dist:
            min_dist = dist
            nearest_category = cat
    return nearest_category, min_dist

# ------------------------------------------------------------------------------
# Color Difference Calculations
# ------------------------------------------------------------------------------

def compute_delta_e(lab1: LABColor, lab2: LABColor) -> float:
    """
    Compute the color difference (ΔE) between two LAB colors using the CIE76 formula.
    
    CIE76 Formula:
      ΔE₇₆ = √[(L₁ - L₂)² + (a₁ - a₂)² + (b₁ - b₂)²]
    
    Note:
      Although simple, CIE76 does not account for perceptual non-uniformities.
    
    Parameters:
        lab1: First LABColor.
        lab2: Second LABColor.
    
    Returns:
        The CIE76 color difference.
    """
    return np.sqrt((lab1.L - lab2.L)**2 + (lab1.a - lab2.a)**2 + (lab1.b - lab2.b)**2)


def compute_delta_e_ciede2000(lab1: LABColor, lab2: LABColor) -> float:
    """
    Compute the color difference (ΔE) between two LAB colors using the CIEDE2000 formula.
    
    The CIEDE2000 formula provides a more accurate representation of perceived color differences 
    compared to CIE76, which is critical in cultural heritage conservation where precise color matching 
    is essential.
    
    Implementation based on Sharma et al. (2005).
    
    Parameters:
        lab1: First LABColor.
        lab2: Second LABColor.
    
    Returns:
        The CIEDE2000 color difference.
    """
    # Unpack LAB components
    L1, a1, b1 = lab1.L, lab1.a, lab1.b
    L2, a2, b2 = lab2.L, lab2.a, lab2.b
    
    # Mean Lightness
    avg_L = (L1 + L2) / 2.0
    
    # Compute chroma for each color
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    
    # Adjust a* values to account for the influence of chroma on hue
    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0
    
    # Compute hue angles in degrees
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    # Compute differences in lightness and chroma
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # Compute delta hue
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        dh = h2_prime - h1_prime
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        delta_h_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh) / 2)
    
    # Calculate mean hue
    if C1_prime * C2_prime == 0:
        avg_h_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            avg_h_prime = (h1_prime + h2_prime) / 2
        elif abs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) < 360:
            avg_h_prime = (h1_prime + h2_prime + 360) / 2
        else:
            avg_h_prime = (h1_prime + h2_prime - 360) / 2
    
    # Compute weighting functions and rotation term
    T = (1 - 0.17 * np.cos(np.radians(avg_h_prime - 30)) +
         0.24 * np.cos(np.radians(2 * avg_h_prime)) +
         0.32 * np.cos(np.radians(3 * avg_h_prime + 6)) -
         0.20 * np.cos(np.radians(4 * avg_h_prime - 63)))
    
    delta_theta = 30 * np.exp(-((avg_h_prime - 275) / 25)**2)
    R_C = 2 * np.sqrt((avg_C_prime**7) / (avg_C_prime**7 + 25**7))
    S_L = 1 + ((0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2))
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    R_T = -np.sin(2 * np.radians(delta_theta)) * R_C
    
    delta_E = np.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_h_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_h_prime / S_H)
    )
    return delta_E

# ------------------------------------------------------------------------------
# Utility functions for display and visualization
# ------------------------------------------------------------------------------

def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app for an enhanced visual layout.
    
    This styling is particularly geared towards presenting technical details in a 
    user-friendly manner for conservation professionals.
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
    
    Displays LAB values, sRGB representation, hexadecimal code, and the nearest 
    ISCC–NBS category, providing comprehensive data for color conservation analysis.
    """
    rgb = color.to_rgb()
    hex_color = color.to_hex()
    iscc_category, dist = find_nearest_iscc_category(color)
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
                    <div style="margin-top: 15px;"><strong>sRGB Values:</strong><br/>RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}<br/>Hex: {hex_color}</div>
                    <div style="margin-top: 15px;"><strong>ISCC–NBS Category:</strong> {iscc_category.title()}<br/><small>(sRGB distance: {dist:.1f})</small></div>
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
    Create a 3D scatter plot to visualize the LAB coordinates of both the original and 
    adjusted colors.
    
    This visualization assists conservation experts in understanding the spatial relationship 
    of colors in the perceptually uniform CIELAB space.
    """
    fig = go.Figure()

    # Plot original color point
    fig.add_trace(go.Scatter3d(
        x=[original.L],
        y=[original.a],
        z=[original.b],
        mode='markers',
        marker=dict(size=8, color=original.to_hex()),
        name=f"Original: {original.name.title()}"
    ))

    # Plot adjusted color point
    fig.add_trace(go.Scatter3d(
        x=[adjusted.L],
        y=[adjusted.a],
        z=[adjusted.b],
        mode='markers',
        marker=dict(size=8, color=adjusted.to_hex()),
        name="Restoration Target"
    ))

    # Draw a connecting line to illustrate the color difference (ΔE)
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
    Provide interactive restoration simulation tools for cultural heritage conservators.
    
    Users can adjust LAB values for a restoration target color and compare it with the 
    original discovered color. Both the simple CIE76 (ΔE₇₆) and the more perceptually 
    accurate CIEDE2000 (ΔE₀₀) color differences are computed for rigorous evaluation.
    """
    st.markdown("## Restoration Simulation Tools")
    st.markdown(
        """
        Adjust the LAB sliders below to fine–tune your restoration target color. The computed 
        color differences quantify how perceptually similar your adjusted color is to the original.
        Both ΔE₇₆ (CIE76) and ΔE₀₀ (CIEDE2000) are provided for a comprehensive analysis.
        """
    )
    # Allow selection among matched colors
    color_names = [color.name.title() for color in matched_colors]
    selected_name = st.selectbox("Select a color for restoration simulation", color_names)
    original_color = next(c for c in matched_colors if c.name.title() == selected_name)

    st.markdown("### Adjust Restoration Target LAB Values")
    target_L = st.slider("Target L (Lightness)", min_value=0.0, max_value=100.0,
                         value=float(original_color.L), step=0.1)
    target_a = st.slider("Target a (Green–Red Axis)", min_value=-128.0, max_value=127.0,
                         value=float(original_color.a), step=0.1)
    target_b = st.slider("Target b (Blue–Yellow Axis)", min_value=-128.0, max_value=127.0,
                         value=float(original_color.b), step=0.1)
    
    adjusted_color = LABColor(name="Restoration Target", L=target_L, a=target_a, b=target_b)

    st.markdown("### Color Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Color**")
        display_color_info(original_color)
    with col2:
        st.markdown("**Restoration Target Color**")
        display_color_info(adjusted_color)
    
    # Compute color differences using both formulas
    delta_e_cie76 = compute_delta_e(original_color, adjusted_color)
    delta_e_ciede2000 = compute_delta_e_ciede2000(original_color, adjusted_color)
    st.markdown(f"### Color Difference (ΔE₇₆ - CIE76): {delta_e_cie76:.2f}")
    st.markdown(f"### Color Difference (ΔE₀₀ - CIEDE2000): {delta_e_ciede2000:.2f}")
    
    st.markdown(
        """
        In controlled conditions, a ΔE value below 2 is usually imperceptible. Values above this 
        threshold may require further restoration adjustments.
        """
    )
    
    st.markdown("### 3D LAB Color Space Visualization")
    plot_lab_3d(original_color, adjusted_color)


def main():
    """
    Main function to run the Streamlit application for color analysis and restoration simulation.
    
    This tool integrates cultural heritage conservation principles with advanced color science,
    enabling detailed visualization and precise evaluation of color differences.
    """
    apply_custom_css()
    
    st.markdown(
        """
        <div class="header-container">
            <h1>MARC LAB Color Visualizer & Restoration Simulator</h1>
            <p>
                Welcome, conservation expert! Upload a MARC (.mrc) file and a CSV with LAB values to visualize 
                the discovered binding color and assess its historical significance. The app assigns an ISCC–NBS 
                category via rigorous sRGB centroid analysis and offers restoration simulation tools for precise color matching.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("📁 File Upload Instructions", expanded=True):
        st.markdown("""
        1. **MARC File**: Upload a .mrc file containing color terms in subfield 'b'.
        2. **CSV File**: Upload a CSV file with columns:
           - Color Name
           - L (Lightness)
           - A (Green–Red)
           - B (Blue–Yellow)
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_marc = st.file_uploader("Upload MARC file (.mrc)", type="mrc")
    with col2:
        uploaded_csv = st.file_uploader("Upload LAB values (CSV)", type="csv")

    if uploaded_marc and uploaded_csv:
        try:
            # Parse MARC file to extract color terms
            record_data = parse_marc_with_pymarc(uploaded_marc.read())
            # Load CSV data for color lookup
            color_manager = ColorDataManager(uploaded_csv)
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
            else:
                st.error("No matching color terms found in the CSV file.")
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
