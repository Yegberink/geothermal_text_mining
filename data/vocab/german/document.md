# German Language Resources

This folder contains German-specific vocabulary and normalization resources used by the workflow.

- `date_locale.yaml`: German month and weekday names for Lexis-style date conversion, including umlaut and ASCII variants such as `März` and `Maerz`.
- `geothermal_patterns.txt`: Regex patterns used for preprocessing geo-hit counts, including `Erdwärme` and `Tiefengeothermie` variants.
- `location_province_overrides.csv`: Optional manual mapping from ambiguous extracted locations to German province/state names. It is currently empty and can be filled with `location,province_name` rows if German geocoding needs manual corrections.
- `keywords_topics.csv`: German frame/category keyword vocabulary used by the paragraph filter and sentence/category classifier.
