
    
    
    
    
    
    
    
    
    
    
    
    
    from pathlib import Path
import gradio as gr
from datetime import datetime
import os
import spaces  # necessary to run on Zero.
from spaces.zero.client import _get_token
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Create a static directory to store the dynamic HTML files
static_dir = Path("./static")
static_dir.mkdir(parents=True, exist_ok=True)

# Tell Gradio which absolute paths are allowed to be served
os.environ["GRADIO_ALLOWED_PATHS"] = str(static_dir.resolve())
print("os.environ['GRADIO_ALLOWED_PATHS'] =", os.environ["GRADIO_ALLOWED_PATHS"])

# Create FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@spaces.GPU(duration=4*60)
def predict(request: gr.Request, text_input):
    token = _get_token(request)
    file_name = f"{datetime.utcnow().strftime('%s')}.html"
    file_path = static_dir / file_name
    print("File will be written to:", file_path)
    with open(file_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-200 dark:text-white dark:bg-gray-900">
    <h1 class="text-3xl font-bold">
        Hello <i>{text_input}</i> From Gradio Iframe
    </h1>
    <h3>Filename: {file_name}</h3>
</body>
</html>
""")
    os.chmod(file_path, 0o644)
    # Use the direct static route instead of Gradio's file route
    iframe = f'<iframe src="/static/{file_name}" width="100%" height="500px"></iframe>'
    link = f'<a href="/static/{file_name}" target="_blank">{file_name}</a>'
    print("Serving file at URL:", f"/static/{file_name}")
    return link, iframe

with gr.Blocks() as block:
    gr.Markdown("""
## Gradio + Static Files Demo
This demo generates dynamic HTML files and stores them in a "static" directory. They are then served via Gradio's `/file=` route.
""")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Name")
            markdown = gr.Markdown(label="Output Link")
            new_btn = gr.Button("New")
        with gr.Column():
            html = gr.HTML(label="HTML Preview", show_label=True)

    new_btn.click(fn=predict, inputs=[text_input], outputs=[markdown, html])

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, block, path="/")

# Run both servers
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)