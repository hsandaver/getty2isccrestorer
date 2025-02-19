import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from skimage import color
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import io
import datetime
import uuid
from abc import ABC, abstractmethod
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import DCTERMS, SKOS, RDF, RDFS
from pymarc import MARCReader

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
# Data Classes and Validation
# ------------------------------------------------------------------------------

@dataclass
class LABColor:
    """
    Represents a color in LAB space.
    """
    name: str
    L: float
    a: float
    b: float

    def __post_init__(self) -> None:
        if not (0 <= self.L <= 100):
            raise ValueError("L must be between 0 and 100.")
        if not (-128 <= self.a <= 127):
            raise ValueError("a must be between -128 and 127.")
        if not (-128 <= self.b <= 127):
            raise ValueError("b must be between -128 and 127.")

    def to_rgb(self) -> Tuple[int, int, int]:
        """Convert LAB to sRGB."""
        lab_array = np.array([[[self.L, self.a, self.b]]])
        rgb_array = color.lab2rgb(lab_array)
        rgb = np.clip(rgb_array[0, 0] * 255, 0, 255).astype(int)
        return tuple(rgb.tolist())

    def to_hex(self) -> str:
        """Return hexadecimal representation of the color."""
        rgb = self.to_rgb()
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

@dataclass
class MARCRecord:
    """
    Represents a simplified MARC record with color terms and metadata.
    """
    color_terms: List[str]
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    oclc: Optional[str] = None

# ------------------------------------------------------------------------------
# Strategy Pattern for Color Difference Calculation
# ------------------------------------------------------------------------------

class ColorDifferenceCalculator(ABC):
    """
    Abstract base class for color difference calculation strategies.
    """
    @abstractmethod
    def calculate(self, lab1: LABColor, lab2: LABColor) -> float:
        pass

class CIE76Calculator(ColorDifferenceCalculator):
    """
    Implements the basic Euclidean distance (CIE76) for delta E.
    """
    def calculate(self, lab1: LABColor, lab2: LABColor) -> float:
        return np.sqrt((lab1.L - lab2.L) ** 2 +
                       (lab1.a - lab2.a) ** 2 +
                       (lab1.b - lab2.b) ** 2)

class CIEDE2000Calculator(ColorDifferenceCalculator):
    """
    Implements the more complex CIEDE2000 delta E calculation.
    """
    def calculate(self, lab1: LABColor, lab2: LABColor) -> float:
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

# ------------------------------------------------------------------------------
# File Parsers and Data Managers
# ------------------------------------------------------------------------------

class MARCProcessor:
    """
    Processes MARC files and extracts color terms and metadata.
    """
    @staticmethod
    def parse(content: bytes) -> MARCRecord:
        try:
            reader = MARCReader(content, to_unicode=True, force_utf8=True)
        except Exception as e:
            raise ValueError(f"Error reading MARC file: {e}")
        
        color_terms: List[str] = []
        oclc: Optional[str] = None
        for record in reader:
            for field in record.get_fields():
                if 'b' in field:
                    b_values = field.get_subfields('b')
                    for value in b_values:
                        term = value.strip().lower()
                        if term:
                            color_terms.append(term)
            # Extract OCLC from field "035", subfield 'a'
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
        return MARCRecord(color_terms=color_terms, oclc=oclc)

class CSVManager:
    """
    Manages CSV file loading and color data extraction.
    """
    def __init__(self, csv_bytes: bytes) -> None:
        self.df = self.load_csv(csv_bytes)
    
    def load_csv(self, csv_bytes: bytes) -> pd.DataFrame:
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

    def find_color(self, color_name: str) -> Optional[LABColor]:
        matched = self.df[self.df['Color Name'] == color_name.lower()]
        if not matched.empty:
            row = matched.iloc[0]
            return LABColor(name=color_name, L=float(row['L']), a=float(row['A']), b=float(row['B']))
        return None

# ------------------------------------------------------------------------------
# Fading Simulator
# ------------------------------------------------------------------------------

class FadingSimulator:
    """
    Simulates color fading under various environmental conditions.
    """
    def __init__(self, dye_type: str) -> None:
        self.dye_type = dye_type.lower()
        if self.dye_type == 'natural':
            self.k = 1e-6  # Natural dyes fade faster
        elif self.dye_type == 'synthetic':
            self.k = 5e-7  # Synthetic dyes are more stable
        else:
            raise ValueError("dye_type must be 'natural' or 'synthetic'")

    def calculate_delta_e(self, original: LABColor, light: float, humidity: float, time: float, temperature: float) -> float:
        """
        Calculate ΔE for fading based on environmental parameters.
        Uses Q10 rule: Q = 2^((temperature - 20)/10)
        """
        Q = 2 ** ((temperature - 20) / 10)
        return self.k * light * humidity * time * Q

    def time_to_fade(self, original: LABColor, light: float, humidity: float, temperature: float, target_delta_e: float = 2.0) -> float:
        """
        Estimate the time (in hours) to reach a target ΔE.
        """
        Q = 2 ** ((temperature - 20) / 10)
        if light * humidity * self.k * Q == 0:
            return float('inf')
        return target_delta_e / (self.k * light * humidity * Q)

def format_time(total_hours: float) -> str:
    """
    Convert time in hours to a string in years, days, hours, and minutes.
    """
    total_minutes = int(round(total_hours * 60))
    years = total_minutes // 525600
    rem_minutes = total_minutes % 525600
    days = rem_minutes // 1440
    rem_minutes %= 1440
    hours = rem_minutes // 60
    minutes = rem_minutes % 60
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return ", ".join(parts) if parts else "0 minutes"

# ------------------------------------------------------------------------------
# RDF Graph Generator
# ------------------------------------------------------------------------------

class RDFGraphGenerator:
    """
    Generates an RDF graph enriched with MARC record and color metadata.
    """
    def __init__(self) -> None:
        self.graph = Graph()
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("skos", SKOS)
        self.graph.bind("lab", LABNS)
        self.graph.bind("ex", EX)

    def add_record(self, record: MARCRecord, colors: List[LABColor]) -> None:
        record_uri = URIRef(f"http://example.org/marc/{record.record_id}")
        self.graph.add((record_uri, RDF.type, EX.MARCRecord))
        self.graph.add((record_uri, RDFS.label, Literal("MARC Record")))
        self.graph.add((record_uri, DCTERMS.created, Literal(datetime.datetime.now().isoformat())))
        if record.oclc:
            self.graph.add((record_uri, DCTERMS.identifier, Literal(record.oclc)))
        for color in colors:
            color_uri = URIRef(f"http://example.org/marc/{record.record_id}/color/{color.name.replace(' ', '_')}")
            self.graph.add((color_uri, RDF.type, EX.ColorTerm))
            self.graph.add((color_uri, SKOS.prefLabel, Literal(color.name.title())))
            # Store computed LAB values
            self.graph.add((color_uri, LABNS.hasL, Literal(color.L)))
            self.graph.add((color_uri, LABNS.hasA, Literal(color.a)))
            self.graph.add((color_uri, LABNS.hasB, Literal(color.b)))
            # Additional triples for the input LAB values
            self.graph.add((color_uri, LABNS.inputL, Literal(color.L)))
            self.graph.add((color_uri, LABNS.inputA, Literal(color.a)))
            self.graph.add((color_uri, LABNS.inputB, Literal(color.b)))
            self.graph.add((color_uri, LABNS.hasHex, Literal(color.to_hex())))
            iscc_category, _ = find_nearest_iscc_category(color)
            self.graph.add((color_uri, SKOS.narrower, Literal(iscc_category.title())))
            self.graph.add((color_uri, LABNS.hasMunsell, Literal(compute_munsell_notation(color))))
            self.graph.add((record_uri, DCTERMS.subject, color_uri))

    def serialize(self, format: str = "turtle") -> str:
        serialized = self.graph.serialize(format=format)
        return serialized.decode("utf-8") if isinstance(serialized, bytes) else str(serialized)

# Helper functions for RDF
def find_nearest_iscc_category(color: LABColor) -> Tuple[str, float]:
    """Finds the nearest ISCC-NBS category based on sRGB distance."""
    rgb = np.array(color.to_rgb(), dtype=float)
    min_dist = float('inf')
    nearest_category = None
    for cat_name, cat_rgb in ISCC_NBS_CENTROIDS.items():
        dist = np.linalg.norm(rgb - np.array(cat_rgb, dtype=float))
        if dist < min_dist:
            min_dist = dist
            nearest_category = cat_name
    return nearest_category, min_dist

def compute_munsell_notation(color: LABColor) -> str:
    """Placeholder for computing Munsell notation."""
    return "5R 4/14"  # Placeholder implementation

# ------------------------------------------------------------------------------
# UI Manager for Streamlit
# ------------------------------------------------------------------------------

class UIManager:
    """
    Manages all UI interactions using Streamlit.
    """
    def __init__(self) -> None:
        self.setup_page()

    def setup_page(self) -> None:
        st.set_page_config(page_title="MARC LAB Color Visualizer & Restoration Simulator", layout="wide")
        st.markdown(
            """
            <div style="text-align:center;">
                <h1>MARC LAB Color Visualizer & Restoration Simulator</h1>
                <p>Welcome, conservation expert! Upload a MARC file and a CSV file with LAB values to begin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def file_upload(self) -> Tuple[Optional[bytes], Optional[bytes]]:
        col1, col2 = st.columns(2)
        with col1:
            marc_file = st.file_uploader("Upload MARC file (.mrc)", type="mrc", help="Select a MARC file.")
        with col2:
            csv_file = st.file_uploader("Upload LAB values CSV", type="csv", help="Select a CSV file with LAB data.")
        return (marc_file.read() if marc_file else None,
                csv_file.read() if csv_file else None)

    def display_color_info(self, color: LABColor) -> None:
        """
        Displays detailed color information including a color swatch.
        """
        rgb = color.to_rgb()
        hex_color = color.to_hex()
        iscc_category, dist = find_nearest_iscc_category(color)
        munsell = compute_munsell_notation(color)
        color_html = f"""
        <div style="background-color:#f7f9fc; border:1px solid #cdd9e5; border-radius:8px; padding:20px; margin:20px 0; box-shadow:0 2px 4px rgba(0,0,0,0.1); font-family:Arial, sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1; padding-right:20px;">
                    <h3>{color.name.title()}</h3>
                    <p><strong>LAB:</strong> L: {color.L:.2f}, a: {color.a:.2f}, b: {color.b:.2f}</p>
                    <p><strong>sRGB:</strong> RGB: {rgb[0]}, {rgb[1]}, {rgb[2]} | Hex: {hex_color}</p>
                    <p><strong>ISCC-NBS:</strong> {iscc_category.title()} (Distance: {dist:.1f})</p>
                    <p><strong>Munsell:</strong> {munsell}</p>
                </div>
                <div style="width:200px; height:200px; background-color:{hex_color}; border:2px solid #333; border-radius:8px;"></div>
            </div>
        </div>
        """
        components.html(color_html, height=300)

    def render_3d_lab(self, original: LABColor, adjusted: LABColor) -> None:
        """
        Renders a 3D LAB color space visualization using Plotly.
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

# ------------------------------------------------------------------------------
# Controller / Main Application Logic
# ------------------------------------------------------------------------------

def main() -> None:
    ui = UIManager()
    marc_bytes, csv_bytes = ui.file_upload()
    
    if marc_bytes and csv_bytes:
        try:
            record = MARCProcessor.parse(marc_bytes)
            csv_manager = CSVManager(csv_bytes)
            
            matched_colors: List[LABColor] = []
            for term in record.color_terms:
                color_obj = csv_manager.find_color(term)
                if color_obj:
                    matched_colors.append(color_obj)
            
            if not matched_colors:
                st.error("No matching color terms found in the CSV file.")
                return
            
            st.markdown("### Matched Colors from MARC File")
            for color_obj in matched_colors:
                ui.display_color_info(color_obj)
            
            with st.expander("Show All Color Terms Found"):
                st.write(", ".join(record.color_terms))
            
            # Restoration Simulation Tools
            st.markdown("## Restoration Simulation Tools")
            color_names = [color.name.title() for color in matched_colors]
            selected_name = st.selectbox("Select a color for restoration simulation", color_names)
            original_color = next((c for c in matched_colors if c.name.title() == selected_name), matched_colors[0])
            
            st.markdown("### Adjust Restoration Target LAB Values")
            custom_lab_input = st.text_input("Enter custom LAB values (L, a, b) separated by commas", value="")
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
                                     value=float(original_color.L), step=0.1)
                target_a = st.slider("Target a (Green–Red)", min_value=-128.0, max_value=127.0,
                                     value=float(original_color.a), step=0.1)
                target_b = st.slider("Target b (Blue–Yellow)", min_value=-128.0, max_value=127.0,
                                     value=float(original_color.b), step=0.1)
            
            adjusted_color = LABColor(name="Restoration Target", L=target_L, a=target_a, b=target_b)
            
            st.markdown("### Color Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Color**")
                ui.display_color_info(original_color)
            with col2:
                st.markdown("**Restoration Target Color**")
                ui.display_color_info(adjusted_color)
            
            # Use strategy pattern to calculate color differences
            cie76 = CIE76Calculator()
            ciede2000 = CIEDE2000Calculator()
            delta_e_cie76 = cie76.calculate(original_color, adjusted_color)
            delta_e_ciede2000 = ciede2000.calculate(original_color, adjusted_color)
            deterioration_msg = "Stable" if delta_e_ciede2000 < 2 else ("Minor deterioration" if delta_e_ciede2000 < 5 else "Significant deterioration")
            
            st.markdown(f"### ΔE (CIE76): {delta_e_cie76:.2f}")
            st.markdown(f"### ΔE (CIEDE2000): {delta_e_ciede2000:.2f}")
            st.markdown(f"### Deterioration Prediction: {deterioration_msg}")
            st.markdown("### 3D LAB Color Space Visualization")
            ui.render_3d_lab(original_color, adjusted_color)
            
            # Fading Simulation
            st.markdown("## Enhanced Fading Simulator")
            fade_color_names = [color.name.title() for color in matched_colors]
            selected_fade_name = st.selectbox("Select a color for fading simulation", fade_color_names, key="fading_simulation")
            original_fade_color = next((c for c in matched_colors if c.name.title() == selected_fade_name), matched_colors[0])
            
            st.markdown("### Environmental Parameters for Fading Simulation")
            light_intensity = st.number_input("Light Intensity (lux)", min_value=0.0, value=5000.0, step=100.0)
            humidity = st.slider("Relative Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
            dye_type = st.selectbox("Dye Type", options=["Natural", "Synthetic"])
            time_exposure = st.slider("Exposure Time (hours)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
            
            fading_simulator = FadingSimulator(dye_type=dye_type)
            computed_delta_e = fading_simulator.calculate_delta_e(original_fade_color, light_intensity, humidity, time_exposure, temperature)
            st.markdown(f"Computed ΔE after {time_exposure} hours: **{computed_delta_e:.2f}**")
            target_time = fading_simulator.time_to_fade(original_fade_color, light_intensity, humidity, temperature)
            st.markdown(f"Estimated time to reach ΔE of 2: **{format_time(target_time)}**")
            
            # RDF Graph Generation
            rdf_generator = RDFGraphGenerator()
            rdf_generator.add_record(record, matched_colors)
            rdf_data = rdf_generator.serialize()
            st.download_button(label="Download RDF Graph (Turtle)",
                               data=rdf_data,
                               file_name="marc_colors.ttl",
                               mime="text/turtle")
        
        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
