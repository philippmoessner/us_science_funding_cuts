# Analyzing US Science Funding Cuts Under the Trump Administration
### Motivation and Research Questions
Since Donald Trump has come into office for his second term as President of the United States of
America (on January 20th, 2025), US-science is experiencing unprecedented funding cuts. These cuts
have potentially wide-reaching consequences for different scientific topics and disciplines, general
development of human potential, public health, economy, and civil society. With this project, we want
to better understand: <br><br>
(a) The effects of the cuts on the scientific landscape:
- Which scientific topics are most affected?
- Which types of institutions are most affected?
- Which states are most affected?
- Are projects in states with specific political party affiliation more affected?
- Are project PIs with specific gender more affected?<br><br>

(b) The potentially underlying procedure or reasoning used to decide on which projects to cut funding:
- Which words appear at higher frequency in affected projects compared to non-affected
projects?
- Can we confirm the use of "banned word lists" to target projects?

### Data
To analyze the effects and procedure of these cuts, we must define a “project population at risk”. This
includes all projects that were active or inactive but receiving money (projects in their closeout phase)
when Trump came into office. We only consider project fundings by the biggest funding agencies for
foundational research – [NIH](https://reporter.nih.gov/exporter) and [NSF](https://www.nsf.gov/awardsearch/download-awards). We complement this by the report of affected projects on [grant-witness.us](https://grant-witness.us/).
<br>
The final enriched dataset (according to the procedure in section 2 and 3 of the `notebook.ipynb`) of NIH and NSF projects since 2014 can be downloaded [here](https://drive.google.com/file/d/19EnZfChrkc0HyIi576zpFVJF3-7JuHvK/view?usp=sharing).

### Results and Visualizations
Currently, the results of the analysis and answers to the specific questions (a) and (b) are presented in a detailed manner in section "4. Analyis" of the `notebook.ipynb` file. Additionally, there are two interactive visualizations available which show the effects of the science funding cuts on the scientific landscape: 
1. [VOS-Viewer Science Map](https://app.vosviewer.com/?json=https%3A%2F%2Fraw.githubusercontent.com%2Fphilippmoessner%2Fus_science_funding_cuts%2Frefs%2Fheads%2Fmain%2Fdata%2FVOSviewer-network_with_weights.json) <br>
Our VOS-Viewer visualization represents different scientific topics as bubbles. Each bubble aggregates information of all corresponding grants / projects and is sized by either the proportion of projects that was cut or the proportion of funding that was lost in this topic (can be toggled via the sidebar). You can view it [here](https://app.vosviewer.com/?json=https%3A%2F%2Fraw.githubusercontent.com%2Fphilippmoessner%2Fus_science_funding_cuts%2Frefs%2Fheads%2Fmain%2Fdata%2FVOSviewer-network_with_weights.json
).<br> [VOS-Viewer](https://github.com/neesjanvaneck/VOSviewer-Online) is an open-source tool for network visualization by Leiden University. Our visualization builds on the base map of OpenAlex topics published [here](https://app.vosviewer.com/?json=https%3A%2F%2Fapp.vosviewer.com%2Fresearchlandscape%2Fdata%2Fopenalex_2023nov.json&url_preview_panel=400).
2. OpenAlex Mapper Science Map <br>
The second visualization displays all grants / projects of the "project population at risk" individually and sizes + colors them by the proportion of budget lost. 
This visualization is available as the `science_map.html` file in this repository and builds on the [OpenAlex Mapper](https://huggingface.co/spaces/MaxNoichl/openalex_mapper) by Maximilian Noichl.
