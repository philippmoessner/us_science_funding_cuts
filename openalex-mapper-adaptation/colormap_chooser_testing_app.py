import gradio as gr
from colormap_chooser import ColormapChooser, setup_colormaps

# Set up colormaps with our preferred collections and ordering
print("Setting up colormaps...")
categories = setup_colormaps(
    included_collections=['matplotlib', 'cmocean', 'scientific', 'cmasher', 'colorbrewer', 'cartocolors'],
    excluded_collections=['colorcet', 'carbonplan', 'sciviz']
)


# Create the chooser with our categories
chooser = ColormapChooser(
    categories=categories,
    smooth_steps=10,
    strip_width=200,
    strip_height=50,


    css_height=180,            # outer box height (becomes a scroll-pane)
    thumb_margin_px=2,         # more space between strips
    gallery_kwargs=dict(columns=3, allow_preview=False, height="200px")   # anything else you need
)

print(chooser.css())

with gr.Blocks(css=chooser.css()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            chooser.render_tabs()
        with gr.Column(scale=2):
            plot = gr.Plot(label="Demo Plot")

    # When the user picks a cmap, update the plot
    def _plot(name):
        print(f"Plotting {name}")
        import numpy as np, matplotlib.pyplot as plt
        data = np.random.RandomState(0).randn(100,100)
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=name)
        fig.colorbar(im, ax=ax)
        plt.close(fig)
        return fig

    chooser.selected_name.change(_plot, chooser.selected_name, plot)

demo.launch(debug=True, share=False, inbrowser=True)
