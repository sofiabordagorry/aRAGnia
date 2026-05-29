import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EN_FILE = DATA_DIR / "openalex_topics.json"
ES_FILE = DATA_DIR / "openalex_topics_es.json"

OUT_DIR = PROJECT_ROOT / "topic_visualizer"
OUT_FILE = OUT_DIR / "merged_topics.js"

def merge_topics():
    print("Leyendo archivos JSON...")
    try:
        with open(EN_FILE, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        with open(ES_FILE, 'r', encoding='utf-8') as f:
            es_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo. Detalles: {e}")
        return

    merged = []
    en_fields = list(en_data.keys())
    es_fields = list(es_data.keys())

    for i in range(len(en_fields)):
        en_field_key = en_fields[i]
        es_field_key = es_fields[i] if i < len(es_fields) else en_field_key
        
        field_id = en_data[en_field_key].get('id', '')
        
        en_subfields_obj = en_data[en_field_key].get('subfields', {})
        es_subfields_obj = es_data.get(es_field_key, {}).get('subfields', en_subfields_obj)
        
        en_subkeys = list(en_subfields_obj.keys())
        es_subkeys = list(es_subfields_obj.keys())
        
        subfields_arr = []
        
        for j in range(len(en_subkeys)):
            en_subkey = en_subkeys[j]
            es_subkey = es_subkeys[j] if j < len(es_subkeys) else en_subkey
            
            en_topics = en_subfields_obj[en_subkey]
            es_topics = es_subfields_obj.get(es_subkey, en_topics)
            
            topics_arr = []
            for k in range(len(en_topics)):
                en_topic = en_topics[k]
                es_topic = es_topics[k] if k < len(es_topics) else en_topic
                topics_arr.append({
                    "en": en_topic,
                    "es": es_topic
                })
            
            subfields_arr.append({
                "nameEn": en_subkey,
                "nameEs": es_subkey,
                "topics": topics_arr
            })
            
        merged.append({
            "id": field_id,
            "nameEn": en_field_key,
            "nameEs": es_field_key,
            "subfields": subfields_arr
        })

    print("Generando archivo JavaScript...")
    json_string = json.dumps(merged, indent=2, ensure_ascii=False)
    
    js_content = f"const mergedData = {json_string};\n"
    
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print(f"¡Éxito! Se ha creado el archivo '{OUT_FILE}'.")

if __name__ == "__main__":
    merge_topics()