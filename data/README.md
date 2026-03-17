## Contents of this folder
1. `data_raw` <br>
This folder includes all raw data files downloaded from [NIH](https://reporter.nih.gov/exporter) and [NSF](https://www.nsf.gov/awardsearch/download-awards) portals after executing the corresponding scripts in `notebook.ipynb`.
Additionally, it contains a data dictionary `NIH_NSF_Data_Dictionary.xlsx` describing the attributes of the datasates from NIH and NSF.

2. `for_enrichment` <br>
This folder contains all data necessary to generate the final enriched dataset. For further information see corresponding section "3.2 Enriching the dataset" in `notebook.ipynb`.

## Final Dataset
The final dataset after merging NIH, NSF, grant-whitness.us data and enriching this merged dataset by crucial context information can be downloaded [here](https://drive.google.com/file/d/19EnZfChrkc0HyIi576zpFVJF3-7JuHvK/view?usp=sharing). This is a MultiIndex pandas dataframe with the following columns: 

| Category | Variable | Description |
| :--- | :--- | :--- |
| **id** | `id` | Unique identifier for the specific research project. |
| **id** | `data_source` | The origin of the project data (e.g., NIH or NSF). |
| **status** | `reinstatement_indicator` | Indicates how a previously terminated project was found to having been reinstated later. |
| **status** | `status` | The funding status of the project. |
| **context** | `ror_match_name` | Name of the institution as identified in the Research Organization Registry (ROR). |
| **context** | `ipeds_edu_type` | Classification of the educational institution according to IPEDS standards. |
| **context** | `ipeds_edu_size` | The size category of the educational institution based on student population. |
| **context** | `ror_inst_type` | Category of organization type assigned by the ROR registry. |
| **context** | `inst_type` | General / derived (from ROR) unique classification of the institution type. |
| **context** | `party_affiliation` | Political party affiliation associated with the project's state. |
| **context** | `gender_inferred` | Inferred gender of the Principal Investigator (PI) based on their first name. |
| **context** | `pi_last_name` | Last name of the Principal Investigator. |
| **context** | `inst_city` | City where the research institution is located. |
| **context** | `inst_state` | State or province where the research institution is located. |
| **context** | `inst_name` | Full name of the research institution receiving the grant. |
| **context** | `inst_country` | Country where the research institution is located. |
| **context** | `pi_first_name` | First name of the Principal Investigator. |
| **time** | `termination_date` | The date on which the project funding was officially terminated. |
| **time** | `project_end` | The originally scheduled end date of the research project. |
| **time** | `rel_prgrss_at_term` | Relative progress of the project duration at the time of termination. |
| **time** | `project_start` | The official start date of the research project. |
| **finance** | `remaining_pct` | The percentage of the total project budget remaining at the time of funding cut / freeze. |
| **finance** | `budget_outlays` | The actual amount of funds disbursed or spent by the project so far. |
| **finance** | `project_budget` | The total authorized budget allocated to the project. |
| **finance** | `outlays_pct` | The percentage of the total budget that has been spent. |
| **finance** | `budget_remaining` | The absolute dollar amount of the budget remaining and not yet spent. |
| **content** | `all_text` | Combined string of the project title and abstract for comprehensive NLP analysis. |
| **content** | `cleaned_tokens` | List of words (lemmata) after lowercase conversion, stop-word removal, and punctuation filtering. |
| **content** | `banned_matched_words` | List of specific keywords from the "banned words" list found within the project text. |
| **content** | `banned_tokens_proportion` | Ratio of the number of matched banned words to the total number of cleaned tokens. |
| **content** | `title` | The original title of the research grant or project. |
| **content** | `abstract` | The original abstract text describing the research goals and methodology. |
| **sci_class** | `domain_id` | Unique identifier for the OpenAlex scientific domain of the research. |
| **sci_class** | `field_name` | The name of the OpenAlex scientific field. |
| **sci_class** | `field_id` | Unique identifier for the OpenAlex scientific field. |
| **sci_class** | `subfield_name` | The name of the OpenAlex scientific subfield. |
| **sci_class** | `subfield_id` | Unique identifier for the OpenAlex scientific subfield. |
| **sci_class** | `topic_id` | Unique identifier for the OpenAlex research topic. |
| **sci_class** | `domain_name` | Name of the OpenAlex scientific domain. |
| **sci_class** | `topic_name` | Name of the OpenAlex research topic classification. |