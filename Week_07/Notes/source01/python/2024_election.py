#data from Wikipedia

import re
import pandas as pd
from io import StringIO

RAW = r"""
Constituency	FF	SF	FG	SD	Lab	II	PBP–S	Aon	GP	Other	Ind	Seats	Quota	Valid ballots	Spoiled ballots	Electorate	Turnout
Carlow–Kilkenny	35.9	17.2	23.6	4.9	2.5	—N/a	2.1	3.4	4.2	1.3	4.9	5	11,627	69,761	563	120,821	58.2
Cavan–Monaghan	27.0	32.8	21.4	—N/a	—N/a	4.4	1.4	7.4	1.1	2.8	1.7	5	11,542	69,246	658	109,152	64.0
Clare	37.8	12.9	24.6	3.5	—N/a	4.2	1.3	3.4	6.2	2.4	3.6	4	12,182	60,907	459	96,398	63.7
Cork East	23.1	13.7	23.1	10.0	—N/a	2.0	1.4	3.7	3.4	0.7	18.9	4	9,602	48,009	317	83,545	57.8
Cork North-Central	23.5	17.4	16.7	5.9	10.2	9.7	5.9	3.2	2.1	4.2	1.2	5	9,846	59,071	552	102,250	58.3
Cork North-West	36.7	12.7	34.8	—N/a	—N/a	2.2	2.0	7.9	2.5	—N/a	1.3	3	10,712	42,844	266	67,255	64.1
Cork South-Central	36.8	15.4	16.2	8.6	4.8	0.7	1.4	3.6	3.4	2.8	7.1	5	10,451	62,704	423	105,076	60.1
Cork South-West	19.3	4.8	23.5	19.9	0.9	23.3	0.7	1.5	0.7	0.6	4.7	3	11,824	47,294	215	74,364	63.9
Donegal	23.9	41.8	9.0	—N/a	—N/a	—N/a	0.8	3.2	1.1	10.2	10.0	5	12,771	76,624	697	131,306	58.9
Dublin Bay North	18.9	19.7	17.2	14.9	4.5	—N/a	3.1	3.8	3.2	4.5	10.4	5	10,929	65,568	524	110,574	59.8
Dublin Bay South	13.9	12.3	25.5	9.1	14.3	—N/a	3.4	2.4	8.2	0.4	10.6	4	7,957	39,784	195	83,689	47.8
Dublin Central	7.2	23.3	16.8	13.3	7.5	—N/a	4.5	2.2	6.0	4.9	14.3	4	6,551	32,754	273	63,190	52.3
Dublin Fingal East	26.0	14.4	14.2	10.6	14.2	—N/a	2.0	4.1	3.6	8.1	2.8	3	9,475	37,899	156	62,465	58.9
Dublin Fingal West	14.7	23.1	15.2	—N/a	16.7	—N/a	4.6	3.9	6.1	3.2	12.4	3	7,530	30,117	168	51,455	58.9
Dublin Mid-West	11.8	32.0	17.9	6.1	4.8	3.8	5.5	3.1	1.3	3.8	9.7	5	7,913	47,476	454	85,769	55.9
Dublin North-West	13.6	30.7	11.9	14.1	2.4	—N/a	8.9	4.2	2.9	4.3	7.0	3	8,184	32,733	300	58,462	56.5
Dublin Rathdown	15.0	7.3	33.7	8.8	6.2	—N/a	2.7	3.7	8.5	0.8	13.3	4	9,752	48,759	241	80,852	60.6
Dublin South-Central	10.7	31.2	8.2	9.0	6.6	2.3	8.9	3.9	6.1	11.0	2.2	4	7,469	37,344	475	77,072	49.1
Dublin South-West	19.5	20.8	19.1	5.9	8.7	—N/a	7.6	3.9	2.9	2.2	9.5	5	11,138	66,823	558	114,832	58.7
Dublin West	22.3	21.0	15.4	4.9	5.5	—N/a	8.0	5.5	6.6	2.6	8.1	5	7,373	44,236	357	78,034	57.1
Dún Laoghaire	15.9	9.0	36.0	7.5	5.7	—N/a	12.2	4.3	7.7	1.2	0.6	4	11,134	55,669	247	95,462	58.6
Galway East	26.2	13.8	21.7	—N/a	—N/a	9.5	2.3	2.9	2.3	1.1	20.3	4	10,843	54,214	362	87,791	62.2
Galway West	16.8	13.5	18.8	3.6	3.3	9.5	1.5	2.0	3.1	1.5	26.4	5	10,047	60,277	375	103,713	58.5
Kerry	26.2	16.3	10.1	—N/a	2.3	1.3	1.3	1.8	2.5	0.9	37.2	5	13,083	78,495	504	120,868	65.4
Kildare North	25.6	14.1	23.3	13.3	5.9	—N/a	2.0	3.8	3.6	3.3	5.1	5	9,505	57,029	357	95,055	60.4
Kildare South	17.4	16.9	21.6	7.4	15.5	2.5	1.2	3.9	1.4	2.2	10.1	4[a]	10,734	42,934	271	74,243	58.2
Laois	22.5	12.8	24.2	—N/a	—N/a	—N/a	1.9	2.7	2.0	—N/a	33.8	3	9,570	38,276	231	65,873	58.5
Limerick City	26.2	17.3	21.2	6.7	6.5	1.6	1.7	4.7	4.2	2.3	7.6	4	8,435	42,174	316	77,753	54.7
Limerick County	24.1	13.2	28.7	—N/a	—N/a	23.1	0.9	2.9	1.9	1.7	3.4	3	11,385	45,536	332	75,018	61.1
Longford–Westmeath	22.4	15.1	29.8	—N/a	2.5	4.8	1.6	3.1	0.7	1.4	18.5	5	10,315	61,884	469	106,814	58.4
Louth	16.8	32.4	15.2	2.4	11.9	0.5	2.6	3.7	2.4	4.2	8.0	5	10,623	63,735	675	104,799	61.5
Mayo	20.0	19.4	35.1	—N/a	—N/a	3.5	1.7	6.3	1.3	—N/a	12.7	5	11,812	70,866	451	112,205	63.6
Meath East	19.3	22.1	26.3	—N/a	4.1	—N/a	2.8	6.6	1.7	2.0	15.3	4	9,997	49,983	338	84,272	59.7
Meath West	17.3	22.8	16.3	6.7	1.1	—N/a	1.3	20.1	1.5	1.7	11.1	3	9,427	37,705	230	65,148	58.2
Offaly	23.8	16.6	17.6	—N/a	—N/a	5.5	1.5	0.8	2.5	—N/a	31.5	3	9,347	37,385	249	62,931	59.8
Roscommon–Galway	17.7	19.5	17.0	—N/a	—N/a	29.2	2.0	1.9	1.1	0.5	11.0	3	10,283	41,128	204	62,727	65.9
Sligo–Leitrim	25.4	22.7	15.8	—N/a	3.7	10.6	2.9	2.0	1.3	2.1	13.6	4	11,381	56,900	416	94,040	60.9
Tipperary North	23.1	9.5	9.7	—N/a	15.5	—N/a	1.4	1.9	1.6	1.0	36.4	3	11,442	45,766	341	70,214	65.7
Tipperary South	14.2	12.0	20.4	—N/a	4.2	0.4	—N/a	1.2	1.4	0.8	45.4	3	10,270	41,079	284	68,247	60.6
Waterford	18.6	33.0	19.3	5.1	2.8	—N/a	1.2	3.1	3.1	0.9	12.9	4	10,731	53,650	363	97,153	55.6
Wexford	16.9	14.5	15.0	—N/a	13.8	—N/a	1.5	7.2	1.4	4.0	25.6	4	10,502	52,508	336	85,744	61.6
Wicklow	6.2	14.8	34.9	13.5	3.5	—N/a	2.2	2.2	4.1	0.4	18.1	4	11,415	57,071	340	84,669	67.8
Wicklow–Wexford	36.1	20.2	23.1	—N/a	—N/a	1.7	2.7	3.5	2.3	—N/a	10.4	3	9,560	38,236	301	63,003	61.2
"""

# --- Parse and clean ---
df = pd.read_csv(StringIO(RAW.strip()), sep="\t")

# Standardise the "—N/a" style entries to NaN
df = df.replace({r"^—.*": pd.NA}, regex=True)

# Columns to use for PCA: first-preference vote shares by party
party_cols = ["FF","SF","FG","SD","Lab","II","PBP–S","Aon","GP","Other","Ind"]

# Convert those to numeric; missing -> 0
for c in party_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# Optional: clean numeric columns with commas (if you want them later)
num_cols = ["Quota","Valid ballots","Spoiled ballots","Electorate","Turnout"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","", regex=False), errors="coerce")

# Seats has a footnote like "4[a]" in one row: strip non-digits
if "Seats" in df.columns:
    df["Seats"] = pd.to_numeric(df["Seats"].astype(str).str.replace(r"\D+", "", regex=True), errors="coerce")

# Save a tidy file for PCA
out_cols = ["Constituency"] + party_cols + ["Turnout"]
df[out_cols].to_csv("irish_ge_firstprefs_clean.csv", index=False)

print(df[out_cols].head())
print("\nWrote: irish_ge_firstprefs_clean.csv")
