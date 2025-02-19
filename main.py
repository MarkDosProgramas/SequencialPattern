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


df_cluster_mean = df.groupby("CLUS_KME")["duration_days"].mean()


tempo_rapido = df_cluster_mean[df_cluster_mean < 3050].index.tolist()
tempo_medio = df_cluster_mean[(df_cluster_mean >= 3050) & (df_cluster_mean <= 4000)].index.tolist()
tempo_lento = df_cluster_mean[df_cluster_mean > 4000].index.tolist()


df_rapidos = df[df["CLUS_KME"].isin(tempo_rapido)]
df_medios = df[df["CLUS_KME"].isin(tempo_medio)]
df_lentos = df[df["CLUS_KME"].isin(tempo_lento)]

print("Clusters Rápidos:", tempo_rapido)
print("Clusters Médios:", tempo_medio)
print("Clusters Lentos:", tempo_lento)


def extract_prefixspan_features(df, suporte_percentual=5):
    sequences = df["event_sequence"].dropna().tolist()
    sequences = [seq for seq in sequences if len(seq) > 1]  
    
    if not sequences:
        return []
    
    suporte_minimo = int(len(sequences) * (suporte_percentual / 100))
    ps = PrefixSpan(sequences)
    ps.minlen = 2  
    ps.minsup = suporte_minimo
    
    patterns = ps.frequent(ps.minsup) 
    extracted_patterns = [{"pattern": pattern, "support": support} for support, pattern in patterns]
    
    return extracted_patterns


padroes_rapidos = extract_prefixspan_features(df_rapidos, suporte_percentual=30)
padroes_medios = extract_prefixspan_features(df_medios, suporte_percentual=40)
padroes_lentos = extract_prefixspan_features(df_lentos, suporte_percentual=30)


def find_exclusive_patterns(patterns_target, patterns_other1, patterns_other2):
    patterns_target_set = {tuple(p["pattern"]) for p in patterns_target}
    patterns_other_set = {tuple(p["pattern"]) for p in patterns_other1} | {tuple(p["pattern"]) for p in patterns_other2}
    
    exclusive_patterns = [p for p in patterns_target if tuple(p["pattern"]) not in patterns_other_set]
    return exclusive_patterns


exclusive_patterns_rapidos = find_exclusive_patterns(padroes_rapidos, padroes_medios, padroes_lentos)
exclusive_patterns_medios = find_exclusive_patterns(padroes_medios, padroes_rapidos, padroes_lentos)
exclusive_patterns_lentos = find_exclusive_patterns(padroes_lentos, padroes_rapidos, padroes_medios)


print("Padrões predominantes EXCLUSIVOS nos Clusters Rápidos:")
for i, pattern in enumerate(exclusive_patterns_rapidos, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")

print("Padrões predominantes nos Clusters Rápidos:")
for i, pattern in enumerate(padroes_rapidos, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")

print("Padrões predominantes EXCLUSIVOS nos Clusters Médios:")
for i, pattern in enumerate(exclusive_patterns_medios, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")

print("Padrões predominantes nos Clusters Médios:")
for i, pattern in enumerate(padroes_medios, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")

print("Padrões predominantes EXCLUSIVOS nos Clusters Lentos:")
for i, pattern in enumerate(exclusive_patterns_lentos, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")

print("Padrões predominantes nos Clusters Lentos:")
for i, pattern in enumerate(padroes_lentos, 1):
    print(f"{i}. Padrão: {pattern['pattern']}, Suporte: {pattern['support']}")