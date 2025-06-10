import streamlit as st
import moderngl
import numpy as np
import subprocess
import json
import os
import sys
import multiprocessing
import signal
import platform
import random
import time
import zipfile
from datetime import datetime
from pathlib import Path

# --- Constants and Setup ---
FFMPEG_PATH = 'ffmpeg' # Default path, user can override in UI

# --- Helper Functions ---

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def safe_popen(*args, **kwargs):
    """ Popen wrapper for cross-platform compatibility """
    if platform.system() == 'Windows':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['preexec_fn'] = os.setsid
    return subprocess.Popen(*args, **kwargs)

def check_required_files():
    """ Check if shader and ffmpeg files exist """
    missing_files = []
    if not os.path.exists(resource_path("shader.vert")):
        missing_files.append("shader.vert")
    if not os.path.exists(resource_path("shader.frag")):
        missing_files.append("shader.frag")
    
    # Check for ffmpeg executable
    try:
        subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_files.append("ffmpeg (Ensure it's in your system's PATH or provide the correct path in the sidebar)")

    return missing_files

# --- Core Logic from rand_params.py ---

PredefinedPalettesData = [
    { "name": "Desert Sundown", "colors": ["#D2691E", "#F4A460", "#FFE4C4", "#8B4513", "#FFF5E1"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Tangerine Dream", "colors": ["#FF7F50", "#FFA500", "#FFEBCD", "#FF4500", "#FF6347"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Avocado Kitchen", "colors": ["#6B8E23", "#A2C523", "#D2B48C", "#8F9779", "#FFF8DC"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Groovy Greens", "colors": ["#98FB98", "#F0FFF0", "#90EE90", "#8FBC8F", "#DCDCDC"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Mustard Pop", "colors": ["#FFDB58", "#F4A460", "#C0C090", "#8B4513", "#F5F5DC"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Burnt Orange Vibe", "colors": ["#FF4500", "#FF6347", "#FFA07A", "#FFD700", "#FFEFD5"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Macrame & Clay", "colors": ["#A0522D", "#DEB887", "#F5DEB3", "#FFB6B3", "#FFEBCD"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Summer Couch", "colors": ["#DAA520", "#FFDAB9", "#8B0000", "#FFE4C4", "#FF7F50"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Olive Groove", "colors": ["#556B2F", "#9ACD32", "#EEE8AA", "#FAFAD2", "#D2B48C"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
    { "name": "Psychedelic Sorbet", "colors": ["#DC143C", "#FF69B4", "#FFA07A", "#FFD700", "#FFF0F5"], "thresholds": [0.2, 0.4, 0.6, 0.8] },
]

def hex_to_rgb_float(hex_color):
    hex_color = hex_color.lstrip('#')
    return [round(int(hex_color[i:i+2], 16) / 255.0, 6) for i in (0, 2, 4)]

def rand_params(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        palette_data = rng.choice(PredefinedPalettesData)
        color_num = len(palette_data["colors"])
        yield {
            "name_prefix": palette_data["name"],
            "shader_parameters": {
                "uStep": 2.0**(rng.randint(-80, -40) / 10),
                "uFrequencyX": rng.randint(1, 20),
                "uFrequencyY": rng.randint(1, 20),
                "uOffset": rng.randint(0, 63) / 10.0,
                "uNoiseInfluence": rng.randint(0, 30) / 10.0,
                "uNoiseScale": rng.randint(0, 30) / 10.0,
                "uNoiseInfluence2": rng.randint(0, 30) / 10.0,
                "uNoiseScale2": rng.randint(0, 30) / 10.0,
                "uOverallScale": rng.randint(0, 30) / 10.0,
                "uThresholds": palette_data["thresholds"] + [2] * (8 - color_num),
                "uColors": [hex_to_rgb_float(c) for c in palette_data["colors"]] + [[0,0,0]] * (8 - color_num)
            }
        }

# --- Core Logic from gen_videos.py ---

VERTEX_SHADER_SOURCE = open(resource_path("shader.vert")).read() if os.path.exists(resource_path("shader.vert")) else ""
FRAGMENT_SHADER_SOURCE = open(resource_path("shader.frag")).read() if os.path.exists(resource_path("shader.frag")) else ""

quad_vertices = np.array([
    -1.0, -1.0, 0.0, 0.0, 1.0,
     1.0, -1.0, 0.0, 1.0, 1.0,
    -1.0,  1.0, 0.0, 0.0, 0.0,
     1.0,  1.0, 0.0, 1.0, 0.0,
], dtype='f4')

def generate_video(params):
    start = time.time()
    width, height, fps, duration, parameters, output_format, output_dir, index, ffmpeg_path, preset, workers_total = params
    shader_parameters = parameters["shader_parameters"]

    try:
        ctx = moderngl.create_standalone_context()
        prog = ctx.program(vertex_shader=VERTEX_SHADER_SOURCE, fragment_shader=FRAGMENT_SHADER_SOURCE)
        vbo = ctx.buffer(quad_vertices.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, 'aPosition', 'aTexCoord')

        y_tex = ctx.texture((width, height), 1, dtype='f1')
        u_tex = ctx.texture((width, height), 1, dtype='f1')
        v_tex = ctx.texture((width, height), 1, dtype='f1')
        fbo = ctx.framebuffer(color_attachments=[y_tex, u_tex, v_tex])
        fbo.use()
        ctx.viewport = (0, 0, width, height)

        for key, value in shader_parameters.items():
            if key in prog:
                prog[key].value = value

        output_path = output_dir / output_format.format(
            name_prefix=parameters.get("name_prefix", "video"),
            width=width, height=height, fps=fps, index=index
        )
        
        threads_per_ffmpeg = max(1, multiprocessing.cpu_count() // workers_total)
        
        ffmpeg_cmd = [
            ffmpeg_path, '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv444p',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '5.2', 
            '-preset', preset, '-threads', str(threads_per_ffmpeg), '-movflags', '+faststart',
            '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            str(output_path),
            '-nostats', '-loglevel', 'error',
        ]

        ffmpeg_subproc = safe_popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        total_frames = int(duration * fps) if duration > 0 else int(2 * np.pi / shader_parameters.get('uStep', 0.01))

        for i in range(total_frames):
            if 'uIndex' in prog:
                prog['uIndex'].value = i
            vao.render(mode=moderngl.TRIANGLE_STRIP)
            for tex in (y_tex, u_tex, v_tex):
                ffmpeg_subproc.stdin.write(tex.read(alignment=1))

        ffmpeg_subproc.stdin.close()
        ffmpeg_subproc.wait()
    except Exception as e:
        return f"Error generating video {index}: {e}"

    end = time.time()
    return f"Generated {output_path.name} in {end - start:.1f}s"

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# --- Streamlit UI ---

st.set_page_config(page_title="Shader Video Generator", layout="wide")

st.title("🎨 Shader Video Generator")
st.markdown("This tool generates videos from a GLSL shader. Configure parameters on the left and start the batch generation.")

# Check for required files at the start
missing = check_required_files()
if missing:
    st.error(f"**Missing Required Files:**\n\n" + "\n".join([f"- `{m}`" for m in missing]) + "\n\nPlease make sure these files are available before starting.")
    st.stop()


# --- Sidebar for Parameters ---
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    param_source = st.radio(
        "Parameter Source",
        ("Randomly Generate", "Upload JSON File"),
        help="Choose to generate new random parameters or upload a specific set from a JSON file."
    )

    st.subheader("Video Properties")
    width = st.number_input("Width (px)", min_value=128, max_value=7680, value=1920, step=1)
    height = st.number_input("Height (px)", min_value=128, max_value=7680, value=1080, step=1)
    fps = st.select_slider("Frames Per Second (FPS)", options=[24, 30, 60, 120], value=60)
    duration = st.number_input("Duration (seconds)", min_value=0.0, value=10.0, step=0.5, help="Set to 0 for a 'perfect loop' based on uStep.")

    st.subheader("Batch Settings")
    if param_source == 'Randomly Generate':
        gen_num = st.number_input("Number of Videos to Generate", min_value=1, max_value=100, value=5, step=1)
        seed = st.number_input("Random Seed", value=123456)
    
    uploaded_file = None
    if param_source == 'Upload JSON File':
        uploaded_file = st.file_uploader("Choose a params.json file", type="json")

    with st.expander("Advanced Settings"):
        st.text_input("FFmpeg Path", value=FFMPEG_PATH, key="ffmpeg_path")
        preset = st.selectbox("FFmpeg Preset", ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], index=5)
        workers = st.slider("Number of Parallel Workers", min_value=1, max_value=multiprocessing.cpu_count(), value=max(1, multiprocessing.cpu_count() // 2))
        output_format = st.text_input("Output Filename Format", value="{name_prefix}-{index:04d}.mp4")

# --- Main Page Logic ---

if st.button("🚀 Start Generating Videos"):
    # Determine parameter list
    parameters_list = []
    if param_source == 'Randomly Generate':
        parameters_list = list(rand_params(gen_num, seed))
    elif uploaded_file is not None:
        try:
            parameters_list = json.load(uploaded_file)
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a correctly formatted file.")
            st.stop()
    
    if not parameters_list:
        st.warning("No parameters loaded. Please generate random parameters or upload a file.")
        st.stop()

    # Create output directory
    now = datetime.now().replace(microsecond=0).isoformat().replace(':', '')
    output_dir_name = f"{now}-{width}x{height}-{fps}fps"
    output_dir = Path("videos") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save params if they were randomly generated
    if param_source == 'Randomly Generate':
        with open(output_dir / "gen-params.json", "w") as f:
            json.dump(parameters_list, f, indent=2)

    # Prepare tasks for multiprocessing
    tasks = [
        (
            width, height, fps, duration,
            params,
            output_format, output_dir, i,
            st.session_state.ffmpeg_path, preset, workers
        )
        for i, params in enumerate(parameters_list)
    ]
    
    st.info(f"Starting generation of {len(tasks)} videos in directory: `{output_dir}`")
    
    results_placeholder = st.empty()
    progress_bar = st.progress(0)
    results = []

    try:
        with multiprocessing.Pool(workers, initializer=init_worker) as pool:
            for i, result in enumerate(pool.imap_unordered(generate_video, tasks)):
                results.append(result)
                progress = (i + 1) / len(tasks)
                progress_bar.progress(progress)
                results_placeholder.info("\n".join(results))

    except KeyboardInterrupt:
        st.warning("Process interrupted by user. Stopping workers.")
        pool.terminate()
        pool.join()
        sys.exit(1)
    
    progress_bar.empty()
    st.success("🎉 Batch generation complete!")
    
    # Display final results and download link
    st.subheader("Results")
    for r in results:
        st.text(r)
        
    # Create a zip file for download
    zip_path = output_dir.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in output_dir.glob('*.mp4'):
            zf.write(file, file.name)
            
    with open(zip_path, "rb") as fp:
        st.download_button(
            label="⬇️ Download All Videos (.zip)",
            data=fp,
            file_name=zip_path.name,
            mime="application/zip"
        )
