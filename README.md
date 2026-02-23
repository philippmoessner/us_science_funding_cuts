# Analyzing the US Science Funding Cuts Under the Trump Administration
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
foundational research – [NIH](https://reporter.nih.gov/exporter) and [NSF](https://www.nsf.gov/awardsearch/download-awards). We complement this by the report of affected projects on [grant-whitness.us](https://grant-witness.us/).
<br>
The final enriched dataset (according to the procedure in section 2 and 3 of the `notebook.ipynb`) of NIH and NSF projects since 2014 can be downloaded [here](https://csh.ac.at/).