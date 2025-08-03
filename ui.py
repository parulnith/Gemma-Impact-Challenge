# Gradio UI and callback logic for the ASHA form app
import gradio as gr
from ai_processing import extract_fields_from_image, transcribe_audio
from config import save_to_csv, image_cache, audio_cache, cache_file, audio_cache_file
from model_utils import load_model  # Use load_model instead of load_model_with_fallback


# Maximum number of form fields supported
MAX_FIELDS = 20


# Main function to build the Gradio UI
def main_ui():
    with gr.Blocks(theme=gr.themes.Soft(), css="""
    /* Apple-like system font stack */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        background: #fff !important; /* white background */
    }
    .skyblue-btn button {
        background: #87ceeb !important;
        color: #1565c0 !important;
        border: none !important;
        font-weight: 600 !important;
        transition: background 0.2s;
    }
    .skyblue-btn button:disabled {
        background: #b3e5fc !important;
        color: #90a4ae !important;
    }
    .skyblue-btn button:hover:not(:disabled) {
        background: #4fc3f7 !important;
        color: #0d47a1 !important;
    }
    .asha-title {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565c0; /* deeper skyblue for title */
        letter-spacing: 0.5px;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .asha-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        font-size: 1.2rem;
        color: #1976d2;
        margin-bottom: 1.2em;
        text-align: center;
    }
    .asha-section { background: #e3f2fd; border-radius: 14px; box-shadow: 0 2px 12px #b3e5fc; padding: 2.2em 2em 1.5em 2em; margin-bottom: 1.5em; }
    .asha-list { font-size: 1.08rem; color: #263238; margin-bottom: 1.2em; }
    .asha-list li { margin-bottom: 0.5em; }
    .asha-note { color: #01579b; font-size: 1.05rem; font-style: italic; margin-top: 1em; }
    .sample-img { border: 2px solid #b3e5fc; border-radius: 8px; margin: 4px; max-width: 120px; cursor: pointer; transition: border 0.2s; }
    .sample-img:hover { border: 2px solid #0288d1; }
    .cancel-audio-x {
        display: inline-block;
        color: #d32f2f;
        font-size: 1.3em;
        font-weight: bold;
        cursor: pointer;
        margin-left: 0.5em;
        vertical-align: middle;
        user-select: none;
        transition: color 0.2s;
    }
    .cancel-audio-x:hover {
        color: #b71c1c;
    }
    /* Icon-style button for cancel audio */
    .cancel-audio-x-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #d32f2f !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        padding: 0 0.3em !important;
        margin-left: 0.5em !important;
        vertical-align: middle !important;
        cursor: pointer !important;
        min-width: 1.7em !important;
        min-height: 1.7em !important;
        border-radius: 50% !important;
        transition: color 0.2s !important;
    }
    .cancel-audio-x-btn button:hover {
        color: #b71c1c !important;
        background: #fbe9e7 !important;
    }
    /* Tab gating styles */
    .tab-disabled {
        opacity: 0.5 !important;
        pointer-events: none !important;
        cursor: not-allowed !important;
    }
    .tab-enabled {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    """) as demo:
        # State variables for UI logic
        fields_state = gr.State([])
        extraction_in_progress = gr.State(False)
        image_uploaded = gr.State(False)
        fields_extracted = gr.State(False)

        # Tabbed interface for workflow steps
        with gr.Tabs() as tabs:
            # About tab: App info and status
            with gr.TabItem("1. About & Demo Info", id=0):
                gr.Markdown("""
# ON DEVICE SMART ASHA Form
### **Digitizing rural health, empowering ASHA workers with AI.**

This application is designed to help **ASHA (Accredited Social Health Activist) workers**—the backbone of India's rural healthcare system—quickly digitize handwritten forms and transcribe Hindi voice input. ASHA workers are often the first point of contact for healthcare in villages, but their work is slowed by manual paperwork and language barriers. This tool streamlines their workflow, making data entry faster, more accurate, and accessible even for those more comfortable with Hindi speech than typing.

- **Image-based field extraction:** Upload a photo of an ASHA form and the app will automatically detect and extract all field labels, ready for digital entry.  
- **Hindi voice transcription:** Fill any field by speaking in Hindi (Devanagari script) for instant, accurate transcription.  
- **Data export:** All submitted data is saved in a CSV for further use or analysis.  

## Powered by Gemma-3n-4B 
This demo runs on the **Gemma 3n 4B model**, a lightweight yet capable model for on-device AI tasks like image-to-text and Hindi speech transcription.  

## On-device CPU Loading
Since this application runs entirely **on-device using CPU**, the model takes some time to **load the first time**. After the initial load, processing is smooth and does not require an internet connection.

## Why this matters
ASHA workers serve over 900 million people in rural India, often with limited digital literacy and resources. By making form digitization and voice transcription seamless, this app saves time, reduces errors, and helps bring rural health data into the digital age—empowering both workers and the communities they serve.

*Note:*
While you are reading this, the **Gemma 3n model** is being loaded in the background to ensure a smooth and fast demo experience. Please explore each step—the workflow is strictly gated for demo clarity. All features are designed for real-world usability and hackathon evaluation.
""")
                model_status = gr.Textbox(value="Loading model in background...", interactive=False, show_label=False, visible=True)

            # Upload tab: Image upload and extraction
            with gr.TabItem("2. Upload Image | छवि अपलोड करें", id=1):
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(type="filepath", label="Upload Form Image | फॉर्म की छवि अपलोड करें", height=350, sources=["upload"])
                        extract_btn = gr.Button("Extract Fields | फ़ील्ड निकालें", variant="secondary", elem_classes=["skyblue-btn"], interactive=False)
                        
                        # Sample images section
                        gr.Markdown("**Or try a sample image | या नमूना छवि आज़माएँ:**", elem_id="sample-image-label")
                        with gr.Row():
                            sample_gallery = gr.Gallery(
                                value=[
                                    "samples/sample_form_2.png",
                                ],
                                label=None,
                                show_label=False,
                                elem_id="sample-gallery",
                                height=110,
                                columns=[3],
                                object_fit="contain",
                                allow_preview=True,
                                interactive=True,
                                elem_classes=["sample-img"]
                            )

            # Fill Form tab: Dynamic form fields and audio input (DISABLED by default)
            with gr.TabItem("3. Fill Form | फॉर्म भरें", id=2, elem_classes=["tab-disabled"]) as fill_form_tab:
                form_placeholder = gr.Markdown("""
## ⚠️ Please extract fields first | कृपया पहले फ़ील्ड निकालें
                
Go to **Step 2** to upload an image and extract form fields before you can fill them.

**कृपया पहले चरण 2 से छवि अपलोड करें और फॉर्म फ़ील्ड निकालें।**
                """, visible=True)
                with gr.Column(visible=False) as form_container:
                    text_inputs, audio_inputs, field_rows = [], [], []
                    for i in range(MAX_FIELDS):
                        with gr.Row(visible=False) as row:
                            text_input = gr.Textbox(interactive=True, label="Enter value | मान दर्ज करें")
                            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Speak to fill | बोलें और भरें", streaming=False)
                            text_inputs.append(text_input)
                            audio_inputs.append(audio_input)
                            field_rows.append(row)
                            # Link audio input to transcription function
                            audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=text_input, show_progress="full")
                
                with gr.Row(visible=False) as action_row:
                    submit_btn = gr.Button("Submit Form | फॉर्म सबमिट करें", variant="primary")
                    new_form_btn = gr.Button("Start New Form | नया फॉर्म शुरू करें")

        # Preload model when app starts
        def preload_model():
            print("[preload_model] Starting model loading...")
            ok = load_model()
            status_msg = "Model loaded and ready! You can now extract form fields." if ok else "Model failed to load. Please check setup and refresh."
            print(f"[preload_model] Result: {status_msg}")
            button_update = gr.update(interactive=True, value="Extract Fields | फ़ील्ड निकालें") if ok else gr.update(interactive=False, value="Model Loading Failed")
            return gr.update(value=status_msg), button_update
        demo.load(fn=preload_model, inputs=None, outputs=[model_status, extract_btn], queue=False)

        # Sample gallery click handler
        def on_sample_click(evt: gr.SelectData):
            img_path = evt.value if isinstance(evt.value, str) else evt.value.get("image", {}).get("path", "")
            print(f"[Sample Click] Selected: {img_path}")
            return gr.update(value=img_path), True

        # Tab gating function
        def enable_fill_form_tab():
            return gr.update(elem_classes=["tab-enabled"])
        
        def disable_fill_form_tab():
            return gr.update(elem_classes=["tab-disabled"])

        # Ensure form fields are cleared when starting new extraction or new form
        def reset_form_fields():
            return [gr.update(value="", visible=False) for _ in text_inputs] + [gr.update(visible=False) for _ in field_rows]

        # Define extract outputs for proper UI updates
        extract_outputs = [extract_btn, form_placeholder, form_container, fields_state, action_row, extraction_in_progress, fill_form_tab] + field_rows + text_inputs
        N = len(extract_outputs)

        # Callback: Extract fields from uploaded image
        def on_extract(img_path):
            def fill_outputs(updates):
                if len(updates) < N:
                    updates += [gr.update()] * (N - len(updates))
                return updates[:N]

            print(f"[on_extract] Called with img_path: {img_path}")
            if not img_path:
                print("[on_extract] No image path provided.")
                yield fill_outputs([
                    gr.update(value="Please upload an image first. | कृपया पहले छवि अपलोड करें।", interactive=True, variant="secondary"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False,  # extraction_in_progress
                    gr.update(elem_classes=["tab-disabled"])  # keep tab disabled
                ])
                return
            
            # Set extraction_in_progress True
            yield fill_outputs([
                gr.update(value="Extracting... | निकाल रहे हैं...", interactive=False, variant="secondary", elem_classes=["skyblue-btn"]),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(),
                gr.update(visible=True),
                True,  # extraction_in_progress
                gr.update(elem_classes=["tab-disabled"])  # keep tab disabled during extraction
            ])
            
            # Check if model is loaded
            if not load_model():
                print("[on_extract] Model failed to load.")
                yield fill_outputs([
                    gr.update(value="Model failed to load. Please refresh and try again. | मॉडल लोड नहीं हुआ। कृपया रीफ्रेश करें।", interactive=True, variant="stop"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False,  # extraction_in_progress
                    gr.update(elem_classes=["tab-disabled"])  # keep tab disabled on error
                ])
                return
            
            try:
                fields = extract_fields_from_image(img_path)
                print(f"[on_extract] extract_fields_from_image returned: {fields}")
            except Exception as e:
                print(f"[on_extract] Exception during extraction: {e}")
                yield fill_outputs([
                    gr.update(value=f"Error during extraction: {str(e)} | निकालने में त्रुटि हुई।", interactive=True, variant="stop"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False,  # extraction_in_progress
                    gr.update(elem_classes=["tab-disabled"])  # keep tab disabled on error
                ])
                return

            if not fields:
                print("[on_extract] No fields found after extraction.")
                yield fill_outputs([
                    gr.update(value="No fields found. | कोई फ़ील्ड नहीं मिली।", interactive=True, variant="stop"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False,  # extraction_in_progress
                    gr.update(elem_classes=["tab-disabled"])  # keep tab disabled if no fields
                ])
                return

            num_fields = min(len(fields), MAX_FIELDS)
            print(f"[on_extract] num_fields to show: {num_fields}")
            print(f"[on_extract] ENABLING Tab 3 - fields successfully extracted!")
            row_updates = [gr.update(visible=i < num_fields) for i in range(MAX_FIELDS)]
            text_updates = [gr.update(label=fields[i] if i < num_fields else "") for i in range(MAX_FIELDS)]
            result = [
                gr.update(value=f"Extracted! | निकाला गया!", interactive=False, variant="secondary", elem_classes=["skyblue-btn"]),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.State(fields),
                gr.update(visible=True),
                False,  # extraction_in_progress
                gr.update(elem_classes=["tab-enabled"])  # ENABLE tab 3 on success!
            ] + row_updates + text_updates
            print(f"[on_extract] Yielding result with {len(result)} outputs.")
            yield fill_outputs(result)

        # Callback: Submit form data to CSV
        def submit_form(*values):
            fields = values[-1]
            text_values = values[:MAX_FIELDS]
            if hasattr(fields, 'value'):
                fields = fields.value
            if not fields:
                return gr.update(value="Error: No fields to submit.", variant="stop", interactive=False), gr.update(visible=True)
            data = dict(zip(fields, text_values[:len(fields)]))
            try:
                save_to_csv(data)
                return gr.update(value="Submitted! | सबमिट हो गया", variant="success", interactive=False), gr.update(visible=True)
            except Exception as e:
                return gr.update(value=str(e), variant="stop", interactive=False), gr.update(visible=True)

        # Callback: Reset form and UI state
        def start_new_form():
            # Reset all form fields and UI state + disable tab 3
            return [
                gr.update(value="Extract Fields | फ़ील्ड निकालें", interactive=True, variant="secondary"),  # reset extract button
                gr.update(visible=True),  # form_placeholder visible
                gr.update(visible=False),  # form_container hidden
                gr.State([]),  # fields_state cleared
                gr.update(visible=False),  # action_row hidden
                False,  # extraction_in_progress
                gr.update(elem_classes=["tab-disabled"])  # DISABLE tab 3 when starting new form
            ] + [gr.update(visible=False) for _ in range(MAX_FIELDS)] + [gr.update(value="", label="") for _ in range(MAX_FIELDS)]

        # Button wiring for UI actions
        # Connect sample gallery to image input
        sample_gallery.select(fn=on_sample_click, inputs=None, outputs=[image_input, image_uploaded])
        
        # Connect new form button to reset function
        new_form_btn.click(
            fn=reset_form_fields,
            inputs=None,
            outputs=text_inputs + field_rows,
            show_progress="minimal"
        )
        extract_btn.click(fn=on_extract, inputs=image_input, outputs=extract_outputs, show_progress="full", concurrency_limit=1)
        submit_btn.click(fn=submit_form, inputs=text_inputs + [fields_state], outputs=[submit_btn, action_row], show_progress="full", concurrency_limit=1)
        new_form_btn.click(fn=start_new_form, inputs=None, outputs=extract_outputs, concurrency_limit=1)
    return demo
