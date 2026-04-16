import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResearcherMention:
    """Mención de investigador en un chunk."""

    name: str
    evidence: str
    chunk_id: str
    cedula: Optional[str] = None
    mail: Optional[str] = None
    afiliacion: Optional[str] = None


@dataclass
class TopicMention:
    """Mención de tópico en un chunk."""

    topic: str
    evidence: str
    chunk_id: str


@dataclass
class LLMExtractionResult:
    """Resultado de extracción LLM."""

    researchers: List[ResearcherMention]
    topics: List[TopicMention]
    errors: List[Dict[str, Any]]


def _extract_json(response: str, chunk_id: str) -> tuple[Any, list]:
    errors: list = []
    # Intentar extraer JSON de tags <JSON>...</JSON>
    json_match = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", response, re.DOTALL)
    data = None
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str), errors
        except json.JSONDecodeError as e:
            errors.append(
                {
                    "type": "JSONDecodeError",
                    "chunk_id": chunk_id,
                    "message": f"JSON dentro de tags inválido: {str(e)}",
                }
            )
            return data, errors
    # Fallback: buscar JSON sin tags (comportamiento anterior)
    json_start = response.find("{")

    if json_start == -1:
        errors.append(
            {
                "type": "InvalidJSON",
                "chunk_id": chunk_id,
                "message": "No hay JSON en respuesta (falta <JSON> tags)",
            }
        )
        return data, errors
    # Intentar encontrar el primer objeto JSON válido
    for json_end in range(len(response), json_start, -1):
        candidate = response[json_start:json_end]
        if candidate.rstrip().endswith("}"):
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        errors.append(
            {
                "type": "JSONDecodeError",
                "chunk_id": chunk_id,
                "message": "No se pudo parsear JSON válido",
            }
        )
    return data, errors


def _generic_name_validation(name: str) -> bool:
    # Validar que el nombre no sea genérico o sin sentido
    invalid_patterns = [
        "nombre completo",
        "nombre del investigador",
        "no se mencionan",
        "no se menciona",
        "investigador",
        "researcher",
        "name",
        "et al",
        "and collaborators",
        "y colaboradores",
        "responsables del grupo",
        "miembros del equipo",
        "equipo de investigación",
        "grupo de investigación",
        "el equipo",
        "los investigadores",
        "el grupo",
        "nuestro grupo",
        "grupo del proyecto",
        "casos similares",
        "contratar",
    ]
    return any(pattern in name.lower() for pattern in invalid_patterns)


def _bibliographic_reference_validation(name: str) -> bool:
    bibliographic_patterns = [
        (r"\bet al\b", 0),  # et al.
        (r"\d{4}\)", 0),  # año entre paréntesis como (2020)
        (r"[A-Z]\.\s*[A-Z]\.", re.IGNORECASE),  # iniciales como J. K.
        (r"^[A-Z]+,\s*[A-Z]\.$", re.IGNORECASE),  # MAHLER, G. o SUESCUN, L.
        (r"^[A-Z]\s+[A-Z]+$", re.IGNORECASE),  # J BREM (inicial + apellido sin puntos)
        (
            r"^[A-Z]\.\s+[A-Z]+",
            re.IGNORECASE,
        ),  # G. SERRA / C. FAGUNDEZ (inicial + apellido)
        (
            r"^[A-Z]\.\s+[A-Z]\.\s+[A-Z]+",
            re.IGNORECASE,
        ),  # J. M. SMITH (dos iniciales + apellido)
    ]
    is_bibliographic = False
    for pattern, flags in bibliographic_patterns:
        if re.search(pattern, name, flags):
            is_bibliographic = True
            break

    return is_bibliographic


def _short_parts_only_validation(name: str) -> bool:

    parts = [part for part in name.split() if part.strip()]
    if not parts:
        return False

    # Sacar conectores comunes para no contar "de", "la", etc.
    stopwords = {"de", "del", "la", "las", "los", "y", "e", "da", "di", "von", "van"}
    meaningful_parts = [p for p in parts if p.lower() not in stopwords]

    if not meaningful_parts:
        return False

    return bool(all(len(p) <= 2 for p in meaningful_parts))


def _surname_only_validation(name: str) -> bool:
    # Detectar: "DEL PUERTO GARCÍA", "NOBOA ALDECOA", etc.
    name_parts = name.split()
    if len(name_parts) >= 2 and all(part.isupper() for part in name_parts):
        # Si todos son mayúsculas y son 2-3 palabras, podría ser solo apellidos
        # Verificar que al menos una parte no sea preposición común
        prepositions = {
            "DE",
            "DEL",
            "LA",
            "LAS",
            "LOS",
            "Y",
            "E",
            "DA",
            "DI",
            "VON",
            "VAN",
        }
        non_prep_parts = [p for p in name_parts if p not in prepositions]

        # Si solo hay 2 partes no-preposición, probablemente son solo apellidos
        return len(non_prep_parts) == 2 and len(name_parts) <= 3
    return False


def _corrupted_characters_validation(name: str) -> bool:
    return bool(re.search(r"[\{\}\[\]\u51fd\u9601\ufffd]", name))


def _biological_species_validation(name: str) -> bool:
    # Patrón: letra mayúscula + punto + palabra (C. elegans, E. granulosus)
    return bool(re.match(r"^[A-Z]\.[\s]?[a-z]+", name))


def _quimical_compound_validation(name: str) -> bool:
    # Patrones: termina con letra mayúscula sola, contiene números/símbolos químicos
    chemical_patterns = [
        r"\b[A-Z]$",  # termina con letra sola como "Aeruciclamida B"
        r"^[A-Z]{2,}$",  # siglas como "DAST"
        r"DIELS.*ALDER",  # reacciones químicas
    ]
    if any(re.search(pattern, name, re.IGNORECASE) for pattern in chemical_patterns):
        name_parts = name.split()
        # Excepción: si contiene espacios y palabras normales, podría ser nombre real
        return not (len(name_parts) >= 2 and any(len(p) > 3 for p in name_parts))
    return False


def _date_validation(name: str) -> bool:
    # Validar que no sea una fecha o mes
    months = {
        "ENERO",
        "FEBRERO",
        "MARZO",
        "ABRIL",
        "MAYO",
        "JUNIO",
        "JULIO",
        "AGOSTO",
        "SEPTIEMBRE",
        "OCTUBRE",
        "NOVIEMBRE",
        "DICIEMBRE",
    }
    return bool(name.upper() in months)


def _institution_or_organization_validation(name: str) -> bool:
    institution_keywords = [
        "CSIC",
        "ANII",
        "DICYT",
        "LABORATORIO",
        "FACULTAD",
        "UNIVERSIDAD",
        "INSTITUTO",
        "CENTRO",
        "DEPARTAMENTO",
        "ACCELERATOR",
        "PROGRAMA",
        "POLO TECNOLÓGICO",
        "ESTUDIANTE",
        "GRUPO I+D",
    ]
    return any(keyword in name.upper() for keyword in institution_keywords)


def _technique_or_section_validation(name: str) -> bool:
    return bool(len(name) > 30 or ("DE " in name.upper() and name.count(" ") > 5))


def _statistic_or_data_validation(name: str) -> bool:
    return bool(re.search(r"\d+\s*%|^[A-Z]\.\s*\d+", name))


def _multiple_names_in_one_validation(name: str) -> bool:
    return bool(";" in name or " and " in name.lower())


def _numbers_in_name_validation(name: str) -> bool:
    parts = name.split()
    return any(re.search(r"\d", part) for part in parts)


def _validate_name(name: str, chunk_id: str) -> List[dict]:
    errors = []

    validation_rules = {
        _multiple_names_in_one_validation: (
            "MultipleNamesInOne",
            "Múltiples nombres en una entidad",
        ),
        _generic_name_validation: ("GenericNameError", "Nombre genérico o sin sentido"),
        _bibliographic_reference_validation: (
            "BibliographicReferenceError",
            "Posible referencia bibliográfica, no participante",
        ),
        _surname_only_validation: ("SurnameOnlyError", "Nombre incompleto (solo apellidos)"),
        _corrupted_characters_validation: (
            "CorruptedCharactersError",
            "Nombre con caracteres corruptos",
        ),
        _biological_species_validation: (
            "BiologicalSpeciesError",
            "Especie biológica, no investigador",
        ),
        _quimical_compound_validation: ("ChemicalCompoundError", "Nombre inválido"),
        _institution_or_organization_validation: (
            "InstitutionError",
            "Institución u organización, no investigador",
        ),
        _technique_or_section_validation: (
            "TechniqueError",
            "Título de sección o técnica, no investigador",
        ),
        _statistic_or_data_validation: ("StatisticError", "Dato estadístico, no investigador"),
        _date_validation: ("DateError", "Mes/fecha, no investigador"),
        _short_parts_only_validation: (
            "ShortPartsOnlyError",
            "Nombre inválido: todas sus partes tienen 2 letras o menos",
        ),
        _numbers_in_name_validation: ("NumbersInNameError", "Nombre inválido: contiene números"),
    }

    for validate, (error_type, message) in validation_rules.items():
        if validate(name):
            errors.append(
                {"type": error_type, "chunk_id": chunk_id, "message": f"{message}: '{name}'."}
            )

    return errors


def _evidence_not_in_chunk_validation(
    evidence: str, chunk_text: str, lenght: int
) -> Optional[float]:
    if chunk_text and evidence and "Mencionado en" not in evidence and len(evidence) > lenght:
        # Normalizar texto: minúsculas y limpiar caracteres de control
        chunk_normalized = _normalize_text(chunk_text)
        evidence_normalized = _normalize_text(evidence)
        # 1. Verificar que la evidencia esté en el chunk
        evidence_words = [
            w for w in re.findall(r"\b\w+\b", evidence_normalized) if len(w) > 3 and not w.isdigit()
        ]
        if len(evidence_words) >= 3:
            words_in_chunk = sum(1 for w in evidence_words if w in chunk_normalized)
            match_ratio = words_in_chunk / len(evidence_words)

            if match_ratio < 0.7:
                return match_ratio
    return None


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _normalize_string(s: str) -> str:
    normalized = unicodedata.normalize("NFD", s)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _clean_name_edges(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^[^\wÀ-ÿ'’-]+", "", name)
    name = re.sub(r"[^\wÀ-ÿ'’-]+$", "", name)
    name = name.strip("_")
    return name


def _match_name(
    name_words: list[str], words_in_chunk: list[str], allowed_errors: int
) -> Optional[list[str]]:
    # recorro todas las palabras del chunk
    for i in range(len(words_in_chunk)):
        errors_used = 0
        matched_words: list = []
        real_name: list = []
        # recorro todas las palabras del nombre
        for name_word in name_words:
            if i >= len(words_in_chunk):
                break
            # cantidad de errores permitidos en la palabra
            allowed_errors_name = max(1, len(name_word) // 3)
            name_normalized = _normalize_string(name_word.strip().lower())
            chunk_word = words_in_chunk[i]
            # Si la palabra del chunk es muy corta, permitir omitirla
            if len(chunk_word) < 3 and len(real_name) > 0:
                while i < len(words_in_chunk) and len(chunk_word) < 3:
                    real_name.append(chunk_word)
                    i += 1
                    if i < len(words_in_chunk):
                        chunk_word = words_in_chunk[i]
                    else:
                        break

            chunk_normalized = _normalize_string(chunk_word.strip().lower())
            distance = _levenshtein_distance(name_normalized, chunk_normalized)
            if distance <= min(allowed_errors_name, allowed_errors - errors_used):
                matched_words.append(chunk_word)
                real_name.append(chunk_word)
                errors_used += distance
                i += 1
            else:
                break

        if len(matched_words) == len(name_words) and errors_used <= allowed_errors:
            return real_name

    return None


def _name_in_chunk_validation(name: str, chunk: str) -> Optional[str]:
    # Cantidad de errores permitidos totales
    allowed_errors = max(1, len(name) // 3)

    # Dividir el chunk en palabras y el nombre
    words_in_chunk = chunk.split()
    name_words = name.split()

    # Intentar hacer coincidir todas las palabras del nombre
    name_true = _match_name(name_words, words_in_chunk, allowed_errors)
    if name_true:
        return _clean_name_edges(" ".join(name_true))

    # Si no se encuentra el nombre, intentar sin palabras de menos de 3 caracteres
    filtered_name_words = [word for word in name_words if len(word) >= 3]
    name_true = _match_name(filtered_name_words, words_in_chunk, allowed_errors)
    if name_true:
        return _clean_name_edges(" ".join(name_true))

    return None


_NULL_STRINGS = {"null", "none", "n/a", "na", "s/d", "no", "no tiene", "", "not mentioned"}

# Cédula uruguaya: 6-8 dígitos, opcionalmente separados con puntos y/o guión verificador
_CEDULA_RE = re.compile(r"^[\d][\d.]{4,9}[-]?\d?$")


def _clean_optional_field(raw: Any) -> Optional[str]:
    """Clean an optional string field from LLM output. Returns None for null-like values."""
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned_lower = cleaned.lower()

    # Exact null-like values
    if cleaned_lower in _NULL_STRINGS:
        return None

    # Null-like values included inside longer strings
    for token in _NULL_STRINGS:
        if token and token in cleaned_lower:
            return None

    if not cleaned:
        return None
    return cleaned


def _validate_cedula(value: str) -> Optional[str]:
    """Validate a Uruguayan cédula (6-8 digits). Format only 8-digit values."""
    # Strip spaces
    value = value.replace(" ", "")
    if not _CEDULA_RE.match(value):
        return None

    digits_only = "".join(c for c in value if c.isdigit())
    # Count actual digits — must be between 6 and 8
    digit_count = len(digits_only)
    if digit_count < 6 or digit_count > 8:
        return None

    if digit_count == 8:
        body, verifier = digits_only[:-1], digits_only[-1]
        body_formatted = re.sub(r"(?<=\d)(?=(\d{3})+$)", ".", body)
        return f"{body_formatted}-{verifier}"

    return value


def _validate_mail(value: str) -> Optional[str]:
    """Validate that a string looks like an email (contains @)."""
    if "@" not in value:
        return None
    return value


def parse_researcher_response(
    response: str, chunk_id: str, chunk_text: str = ""
) -> LLMExtractionResult:
    """Parsear respuesta del LLM."""
    errors: list = []
    researchers: list = []
    data, errors = _extract_json(response, chunk_id)
    if errors != []:
        return LLMExtractionResult(researchers=[], topics=[], errors=errors)
    try:
        researchers_data, errors = _valid_json_structure(data, "researchers", chunk_id)
        if errors != []:
            return LLMExtractionResult(researchers=[], topics=[], errors=errors)

        for item in researchers_data:
            name, evidence, errors_aux = _validate_json_sub_structure(item, "name", chunk_id)
            if errors_aux != []:
                errors.extend(errors_aux)
                continue

            if not name:
                errors.append(
                    {
                        "type": "MissingName",
                        "chunk_id": chunk_id,
                        "message": "Falta el campo 'name' o está vacío",
                    }
                )
                continue

            # 2. Verificar que el nombre esté en el chunk
            real_name = _name_in_chunk_validation(name, chunk_text)
            if real_name:
                name = real_name

            errors_aux = _validate_name(name, chunk_id)
            if errors_aux:
                errors.extend(errors_aux)
                continue

            # VALIDACIÓN: Verificar que evidencia y nombre estén en el chunk
            match_ratio = _evidence_not_in_chunk_validation(evidence, chunk_text, 1)
            if match_ratio is not None:
                errors.append(
                    {
                        "type": "EvidenceNotInChunk",
                        "chunk_id": chunk_id,
                        "message": f"La evidencia '{evidence[:80]}...' no está en el chunk (solo {match_ratio:.0%} de palabras coinciden)",
                    }
                )
                continue

            # Extraer propiedades opcionales
            cedula = _clean_optional_field(item.get("cedula"))
            if cedula:
                cedula = _validate_cedula(cedula)

            mail = _clean_optional_field(item.get("mail"))
            if mail:
                mail = _validate_mail(mail)

            afiliacion = _clean_optional_field(item.get("afiliacion"))

            # Si llegamos hasta acá, pasó todas las validaciones
            researchers.append(
                ResearcherMention(
                    name=name,
                    evidence=evidence,
                    chunk_id=chunk_id,
                    cedula=cedula,
                    mail=mail,
                    afiliacion=afiliacion,
                )
            )

    except json.JSONDecodeError as e:
        errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

    return LLMExtractionResult(researchers=researchers, topics=[], errors=errors)


def _list_topic_validation(topic: str, available_topics: list[str]) -> bool:
    # Validar que el tópico esté en la lista permitida
    # Comparación case-insensitive
    topic_lower = topic.lower()
    available_lower = [t.lower() for t in available_topics]

    return topic_lower not in available_lower


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\t\r\n]+", " ", text.lower()))


def _valid_json_structure(data: Any, info, chunk_id: str) -> tuple[list, list]:
    info_data = data.get(info, [])
    errors = []
    if not isinstance(info_data, list):
        errors.append(
            {"type": "InvalidJSON", "chunk_id": chunk_id, "message": "'topics' no es lista"}
        )
    return info_data, errors


def _validate_json_sub_structure(item: Any, value: str, chunk_id: str) -> tuple[str, str, list]:
    errors = []
    raw_value = ""
    evidence = ""
    if not isinstance(item, dict):
        errors.append(
            {
                "type": "InvalidJSON",
                "chunk_id": chunk_id,
                "message": "Elemento de 'topics' no es un objeto",
            }
        )
        return raw_value, evidence, errors
    # Manejar casos donde el valor puede ser lista, None, u otro tipo
    raw = item.get(value, "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    raw_value = str(raw).strip() if raw else ""

    raw_evidence = item.get("evidence", "")
    if isinstance(raw_evidence, list):
        raw_evidence = raw_evidence[0] if raw_evidence else ""
    evidence = str(raw_evidence).strip() if raw_evidence else f"Mencionado en {chunk_id}"
    return raw_value, evidence, errors


def parse_topic_response(
    response: str, available_topics: list[str], chunk_id: str, chunk_text: str = ""
) -> LLMExtractionResult:
    """Parsear respuesta del LLM para tópicos."""
    errors: list = []
    topics: list = []

    data, errors = _extract_json(response, chunk_id)
    if errors != []:
        return LLMExtractionResult(researchers=[], topics=[], errors=errors)

    try:
        topics_data, errors = _valid_json_structure(data, "topics", chunk_id)
        if errors != []:
            return LLMExtractionResult(researchers=[], topics=[], errors=errors)

        for item in topics_data:
            topic, evidence, errors_aux = _validate_json_sub_structure(item, "topic", chunk_id)
            if errors_aux != []:
                errors.extend(errors_aux)
                continue

            if topic is None:
                errors.append(
                    {
                        "type": "MissingTopic",
                        "chunk_id": chunk_id,
                        "message": "Falta el campo 'topic' o está vacío",
                    }
                )
                continue

            # Validar que el tópico esté en la lista permitida
            if _list_topic_validation(topic, available_topics):
                errors.append(
                    {
                        "type": "InvalidTopic",
                        "chunk_id": chunk_id,
                        "message": f"Tópico '{topic}' no está en la lista permitida (inventado por LLM)",
                    }
                )
                continue

            # VALIDACIÓN: La evidencia debe estar en el chunk original
            match_ratio = _evidence_not_in_chunk_validation(evidence, chunk_text, 15)
            if match_ratio is not None:
                errors.append(
                    {
                        "type": "EvidenceNotInChunk",
                        "chunk_id": chunk_id,
                        "message": f"La evidencia '{evidence[:80]}...' no está en el chunk (solo {match_ratio:.0%} de palabras coinciden)",
                    }
                )
                continue
            topics.append(TopicMention(topic=topic, evidence=evidence, chunk_id=chunk_id))

    except json.JSONDecodeError as e:
        errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

    return LLMExtractionResult(researchers=[], topics=topics, errors=errors)
