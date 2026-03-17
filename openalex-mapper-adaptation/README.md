[OpenAlex Mapper](https://huggingface.co/spaces/MaxNoichl/openalex_mapper) is a tool developed by Maximilian Noichl. At its core it allows to provide abstract and title information of scientific projects, embeds them into latent word-vector representations and maps these embeddings back onto a 2D visualization. This visualization serves as a "Map of Science" where the provided scientific projects are located.

The public version of this tool only allows to provide lists of existing scientific projects / publications queried via the OpenAlex search engine. Since our data refers to unpublished / ongoing projects, this restriction of the tool had to be overcome. We therefore cloned the repository and implemented several additional features to make the tool usable for our data. These are the newly implemented features:
- Added support for uploading your own CSV datasets (not just OpenAlex query results).
- Added dynamic Plot Coloring Type options from extra uploaded CSV columns.
- Enabled coloring by custom columns (both numeric and categorical), including automatic legend generation.
- Added bubble-size scaling based on selected numeric measure (with baseline preserved and exponential scaling for stronger visual differences).
- Added selected-measure value to point hover tooltips.
- Added embedding caching (persisted on disk) so already-seen records can reuse embeddings across runs.
- Improved handling of standalone interactive HTML export.

Please consider using the [original version by Maximilian Noichl](https://huggingface.co/spaces/MaxNoichl/openalex_mapper). All credits go to him.