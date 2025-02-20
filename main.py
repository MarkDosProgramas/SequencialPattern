import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from datetime import datetime
from prefixspan import PrefixSpan

xes_log = xes_importer.apply("/root/projeto_byronv2/datasets/TJ_clustered.xes")

data = []

for trace in xes_log:
    trace_data = {key: value for key, value in trace.attributes.items()}
    trace_data["concept:name"] = trace.attributes.get("concept:name", "Unknown")
    
    if len(trace) > 0:
        start_time = trace[0]["time:timestamp"]
        end_time = trace[-1]["time:timestamp"]
        duration_days = (end_time - start_time).total_seconds() / (24 * 3600)
        trace_data["duration_days"] = duration_days
    else:
        trace_data["duration_days"] = None
    
    trace_data["event_sequence"] = [event["concept:name"] for event in trace]
    
    data.append(trace_data)

df = pd.DataFrame(data)

if "CLUS_KME" in df.columns:
    grupos = sorted(df["CLUS_KME"].unique())
    grupos_map = {y: x for x, y in enumerate(grupos)}
    df['CLUS_KME'] = df['CLUS_KME'].map(grupos_map)
else:
    raise ValueError("Erro: A coluna 'CLUS_KME' não foi encontrada.")

df_cluster_mean = df.groupby("CLUS_KME")["duration_days"].mean().sort_values()


print(df_cluster_mean)

tempo_rapido = df_cluster_mean[df_cluster_mean < 2443.06].index.tolist()
tempo_medio = df_cluster_mean[(df_cluster_mean >= 2443.06 ) & (df_cluster_mean <= 4141.02 )].index.tolist()
tempo_lento = df_cluster_mean[df_cluster_mean > 4141.02 ].index.tolist()

df_rapidos = df[df["CLUS_KME"].isin(tempo_rapido)]
df_medios = df[df["CLUS_KME"].isin(tempo_medio)]
df_lentos = df[df["CLUS_KME"].isin(tempo_lento)]

def extract_prefixspan_features(df, suporte_percentual=5):
    sequences = df["event_sequence"].dropna().tolist()
    sequences = [seq for seq in sequences if len(seq) > 1]
    
    if not sequences:
        return []
    
    total_sequences = len(sequences)
    suporte_minimo = int(total_sequences * (suporte_percentual / 100))
    ps = PrefixSpan(sequences)
    ps.minlen = 2
    ps.minsup = suporte_minimo
    
    patterns = ps.frequent(ps.minsup)
    extracted_patterns = [
        {"pattern": pattern, "support": (support / total_sequences) * 100}
        for support, pattern in patterns
    ]
    
    return sorted(extracted_patterns, key=lambda x: x["support"], reverse=True)

padroes_rapidos = extract_prefixspan_features(df_rapidos, suporte_percentual=40)
padroes_medios = extract_prefixspan_features(df_medios, suporte_percentual=65)
padroes_lentos = extract_prefixspan_features(df_lentos, suporte_percentual=53)

def find_exclusive_patterns(patterns_target, patterns_other1, patterns_other2, limiar=35):
    patterns_target_dict = {tuple(p["pattern"]): p["support"] for p in patterns_target}
    patterns_other_dict = {}
    
    for p in patterns_other1 + patterns_other2:
        pattern_tuple = tuple(p["pattern"])
        if pattern_tuple in patterns_other_dict:
            patterns_other_dict[pattern_tuple] += p["support"]
        else:
            patterns_other_dict[pattern_tuple] = p["support"]
    
    exclusive_patterns = []
    for pattern, support in patterns_target_dict.items():
        other_support = patterns_other_dict.get(pattern, 0)
        if support > other_support + limiar:
            exclusive_patterns.append({"pattern": list(pattern), "support": support})
    
    return sorted(exclusive_patterns, key=lambda x: x["support"], reverse=True)

exclusive_patterns_rapidos = find_exclusive_patterns(padroes_rapidos, padroes_medios, padroes_lentos)
exclusive_patterns_medios = find_exclusive_patterns(padroes_medios, padroes_rapidos, padroes_lentos)
exclusive_patterns_lentos = find_exclusive_patterns(padroes_lentos, padroes_rapidos, padroes_medios)

with open("resultados.txt", "w", encoding="utf-8") as f:
    def print_patterns(title, patterns):
        f.write(title + "\n")
        for i, pattern in enumerate(patterns, 1):
            f.write(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']:.2f}%\n")
        f.write("\n")

    print_patterns("Padrões predominantes EXCLUSIVOS nos Clusters Rápidos:", exclusive_patterns_rapidos)
    print_patterns("Padrões predominantes nos Clusters Rápidos:", padroes_rapidos)
    print_patterns("Padrões predominantes EXCLUSIVOS nos Clusters Médios:", exclusive_patterns_medios)
    print_patterns("Padrões predominantes nos Clusters Médios:", padroes_medios)
    print_patterns("Padrões predominantes EXCLUSIVOS nos Clusters Lentos:", exclusive_patterns_lentos)
    print_patterns("Padrões predominantes nos Clusters Lentos:", padroes_lentos)
