import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from skimage import color
from dataclasses import dataclass
from typing import Optional, Tuple, List
import re
import io
from pymarc import MARCReader  # Ensure pymarc is installed (pip install pymarc)


@dataclass
class LABColor:
    name: str
    L: float
    a: float
    b: float

    def to_rgb(self) -> Tuple[int, int, int]:
        """Convert LAB values to RGB using the CIELAB color space."""
        lab_array = np.array([[[self.L, self.a, self.b]]])
        rgb_array = color.lab2rgb(lab_array)
        return tuple((rgb_array[0, 0] * 255).astype(int))

    def to_hex(self) -> str:
        """Convert LAB values to a hex color code."""
        rgb = self.to_rgb()
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


@dataclass
class MARCRecord:
    color_terms: List[str]


def parse_marc_with_pymarc(content: bytes) -> MARCRecord:
    """
    Parse the MARC file using pymarc and extract color terms.
    This version ignores title and OCLC number.
    """
    reader = MARCReader(content, to_unicode=True, force_utf8=True)
    color_terms = []

    # Process only the first record in the file
    for record in reader:
        # Extract color terms from all fields that contain subfield 'b'
        for field in record.get_fields():
            if 'b' in field:
                # Get all 'b' subfields
                b_values = field.get_subfields('b')
                for value in b_values:
                    color_terms.append(value.strip().lower())
        break  # Only process the first record

    return MARCRecord(color_terms=color_terms)


class ColorDataManager:
    def __init__(self, csv_data: io.BytesIO):
        self.df = self._load_csv(csv_data)
    
    @staticmethod
    def _load_csv(csv_data: io.BytesIO) -> pd.DataFrame:
        """Load and validate CSV data containing color reference information."""
        df = pd.read_csv(csv_data)
        required_columns = {'Color Name', 'L', 'A', 'B'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        df['Color Name'] = df['Color Name'].str.lower().str.strip()
        return df
    
    def find_color(self, color_name: str) -> Optional[LABColor]:
        """Find LAB values for a given color name."""
        color_row = self.df[self.df['Color Name'] == color_name.lower()]
        if not color_row.empty:
            return LABColor(
                name=color_name,
                L=color_row.iloc[0]['L'],
                a=color_row.iloc[0]['A'],
                b=color_row.iloc[0]['B']
            )
        return None


def apply_custom_css():
    """Apply custom styling to the main app container."""
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
    """Render the color information and swatch using an HTML component."""
    rgb = color.to_rgb()
    hex_color = color.to_hex()
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
                    <div style="margin-top: 15px;"><strong>Color Values:</strong><br/>RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}<br/>Hex: {hex_color}</div>
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


def compute_delta_e(lab1: LABColor, lab2: LABColor) -> float:
    """
    Compute the color difference (ΔE) between two LAB colors using the CIE76 formula.
    ΔE₇₆ = √[(L₁ - L₂)² + (a₁ - a₂)² + (b₁ - b₂)²]
    """
    return np.sqrt((lab1.L - lab2.L)**2 + (lab1.a - lab2.a)**2 + (lab1.b - lab2.b)**2)


def plot_lab_3d(original: LABColor, adjusted: LABColor):
    """
    Create a 3D scatter plot of LAB coordinates for both the original and adjusted colors.
    This visualization aids in understanding the color's location within the CIELAB color space.
    """
    fig = go.Figure()

    # Original color point
    fig.add_trace(go.Scatter3d(
        x=[original.L],
        y=[original.a],
        z=[original.b],
        mode='markers',
        marker=dict(size=8, color=original.to_hex()),
        name=f"Original: {original.name.title()}"
    ))

    # Adjusted (target) color point
    fig.add_trace(go.Scatter3d(
        x=[adjusted.L],
        y=[adjusted.a],
        z=[adjusted.b],
        mode='markers',
        marker=dict(size=8, color=adjusted.to_hex()),
        name="Adjusted/Restoration Target"
    ))

    # Connecting line to illustrate ΔE
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
            yaxis_title="a (Green-Red)",
            zaxis_title="b (Blue-Yellow)"
        ),
        title="CIELAB Color Space Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)


def restoration_tools(matched_colors: List[LABColor]):
    """
    Provide an interactive interface for conservators to simulate restoration adjustments.
    The user can select one of the matched colors and adjust LAB values to achieve the desired restoration color.
    """
    st.markdown("## Restoration Simulation Tools")
    st.markdown(
        """
        The following tools allow you to compare the discovered binding color with a proposed restoration color.
        Adjust the LAB sliders to simulate the changes required in the restoration process. The computed ΔE value 
        quantifies the color difference. In conservation science, a lower ΔE indicates a closer match between the 
        original and the restoration target, thereby guiding the selection of pigments or corrective measures.
        """
    )
    # Allow conservator to select one of the found colors (if multiple)
    color_names = [color.name.title() for color in matched_colors]
    selected_name = st.selectbox("Select a color for restoration simulation", color_names)
    original_color = next(c for c in matched_colors if c.name.title() == selected_name)

    st.markdown("### Adjust Restoration Target LAB Values")
    # Sliders to adjust the target LAB values
    target_L = st.slider("Target L (Lightness)", min_value=0.0, max_value=100.0, value=float(original_color.L), step=0.1)
    target_a = st.slider("Target a (Green-Red Axis)", min_value=-128.0, max_value=127.0, value=float(original_color.a), step=0.1)
    target_b = st.slider("Target b (Blue-Yellow Axis)", min_value=-128.0, max_value=127.0, value=float(original_color.b), step=0.1)
    
    adjusted_color = LABColor(name="Restoration Target", L=target_L, a=target_a, b=target_b)

    # Display both original and adjusted color info
    st.markdown("### Color Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Color**")
        display_color_info(original_color)
    with col2:
        st.markdown("**Restoration Target Color**")
        display_color_info(adjusted_color)
    
    # Compute and display ΔE difference
    delta_e = compute_delta_e(original_color, adjusted_color)
    st.markdown(f"### Color Difference (ΔE₇₆): {delta_e:.2f}")
    
    st.markdown(
        """
        A ΔE value less than 2 is generally considered imperceptible to the human eye under controlled conditions.
        Higher values indicate more significant discrepancies, which may necessitate further adjustment of the restoration protocol.
        """
    )
    
    # Provide a 3D visualization of the color shift in LAB space
    st.markdown("### 3D LAB Color Space Visualization")
    plot_lab_3d(original_color, adjusted_color)


def main():
    apply_custom_css()
    
    st.markdown(
        """
        <div class="header-container">
            <h1>MARC LAB Color Visualizer & Restoration Simulator</h1>
            <p>
                Upload a MARC (.mrc) file and a CSV with LAB values to visualize the binding color discovered in the MARC file.
                This tool also offers interactive restoration simulation for conservators seeking to replicate or restore the exact hue.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("📁 File Upload Instructions", expanded=True):
        st.markdown("""
        1. **MARC File**: Upload a .mrc file containing color terms in subfield 'b'.
        2. **CSV File**: Upload a CSV file with the following columns:
           - Color Name
           - L (Lightness value)
           - A (Green-Red value)
           - B (Blue-Yellow value)
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_marc = st.file_uploader("Upload MARC file (.mrc)", type="mrc")
    with col2:
        uploaded_csv = st.file_uploader("Upload LAB values (CSV)", type="csv")

    if uploaded_marc and uploaded_csv:
        try:
            # Parse the MARC file for color terms using pymarc
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
                
                # Invoke the restoration simulation tools for further analysis
                restoration_tools(matched_colors)
            else:
                st.error("No matching color terms found in the CSV file.")
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()