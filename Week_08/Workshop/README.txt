PCA workshop: the geography of British dialect words
====================================================

Files
-----
dialect_pca_workshop.ipynb          student notebook (explanation, exercises, questions)
dialect_pca_workshop_worked.ipynb   the same notebook with code, outputs and answers filled in
bbc_voices.csv                      124 postcode areas x 139 words, % within each concept (BBC Voices survey)
twitter.csv                         same shape, from geolocated UK tweets (2014)
areas.csv                           postcode area code, name, nation, centre (lat, lon), approx_respondents
survey_questions.csv                the 36 concepts and the words recorded for each
postcode_areas.json                 simplified outline of each postcode area, for the maps
dialect_maps.py                     plot_map and plot_rgb_map

Keep all the files in one folder and open the notebook from there.
Needs Python 3 with numpy, pandas, matplotlib, scikit-learn and scipy.

Where the data come from
------------------------
The two percentage tables are the supplementary data of
  Grieve, Montgomery, Nini, Murakami and Guo (2019), Mapping lexical dialect
  variation in British English using Twitter, Frontiers in Artificial
  Intelligence 2:11, doi 10.3389/frai.2019.00011 (CC BY 4.0).
The survey figures in that paper derive from the BBC Voices survey (2004-05).

The outlines come from github.com/missinglink/uk-postcode-polygons, which is an
export of the postcode-district maps on Wikipedia; districts were merged into
areas and simplified. Northern Ireland (BT), Guernsey, Jersey and the Isle of
Man have no outline there and are drawn as circles.

What was done to the data
-------------------------
- bbc_voices.csv: eight areas had no answers at all for one or more concepts
  (48 cells). Those cells were filled with the average over all areas.
- areas.csv, approx_respondents: the source files do not give the number of
  respondents. It was estimated, for each area, as the smallest number of
  people consistent with the rounded percentages of each concept, and the
  median over the 36 concepts is reported. Treat it as approximate.
- The wording of the 36 concepts in survey_questions.csv is a paraphrase of
  the BBC Voices prompts.
