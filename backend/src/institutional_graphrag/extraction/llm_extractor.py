"""Extracción de investigadores usando LLMs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from institutional_graphrag.extraction.parse_response import parse_researcher_response, parse_topic_response, LLMExtractionResult

from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PARTICIPO_EN,
    Entity,
    Investigador,
    Relationship,
    Topico,
)
from institutional_graphrag.rag.generate import get_llm_client

logger = logging.getLogger(__name__)

# Lista de subfields de OpenAlex para guiar la extracción de tópicos
OPENALEX_TOPICS = [
    "Sociology and Political Science",
    "Nuclear and High Energy Physics",
    "Plant Science",
    "Molecular Biology",
    "Electrical and Electronic Engineering",
    "Artificial Intelligence",
    "Political Science and International Reations",
    "Economics and Econometrics",
    "Education",
    "Aerospace Engineering",
    "Surgery",
    "Materials Chemistry",
    "History",
    "Astronomy and Astrophysics",
    "Biomedical Engineering",
    "General Health Professions",
    "Atomic and Molecular Physics, and Optics",
    "Philosophy",
    "Mechanical Engineering",
    "Information Systems",
    "Public Health, Environmental and Occupational Health",
    "Literature and Literary Theory",
    "Ecology, Evolution, Behavior and Systematics",
    "Pediatrics, Perinatology and Child Health",
    "Genetics",
    "Epidemiology",
    "Pulmonary and Respiratory Medicine",
    "Ecology",
    "Organic Chemistry",
    "Archeology",
    "Clinical Psychology",
    "Civil and Structural Engineering",
    "Control and Systems Engineering",
    "Computer Networks and Communications",
    "Strategy and Management",
    "Law",
    "Language and Linguistics",
    "Anthropology",
    "Social Psychology",
    "Management, Monitoring, Policy and Law",
    "Oncology",
    "Physiology",
    "Cardiology and Cardiovascular Medicine",
    "Computer Vision and Pattern Recognition",
    "Global and Planetary Change",
    "Radiology, Nuclear Medicine and Imaging",
    "Cultural Studies",
    "Mechanics of Materials",
    "Computational Mechanics",
    "Food Science",
    "Pharmacology",
    "Cognitive Neuroscience",
    "Accounting",
    "Geophysics",
    "Atmospheric Science",
    "Ocean Engineering",
    "Infectious Diseases",
    "Immunology",
    "Demography",
    "Water Science and Technology",
    "Renewable Energy, Sustainability and the Environment",
    "Computational Theory and Mathematics",
    "Biophysics",
    "Religious studies",
    "Endocrinology, Diabetes and Metabolism",
    "Oceanography",
    "Pathology and Forensic Medicine",
    "Paleontology",
    "Condensed Matter Physics",
    "Management Science and Operations Research",
    "Environmental Engineering",
    "History and Philosophy of Science",
    "Building and Construction",
    "Insect Science",
    "Finance",
    "Information Systems and Management",
    "Organizational Behavior and Human Resource Management",
    "Neurology",
    "Experimental and Cognitive Psychology",
    "Biomaterials",
    "Psychiatry and Mental health",
    "Nature and Landscape Conservation",
    "Urban Studies",
    "Visual Arts and Performing Arts",
    "Developmental and Educational Psychology",
    "Industrial and Manufacturing Engineering",
    "Health, Toxicology and Mutagenesis",
    "Rheumatology",
    "Cell Biology",
    "Reproductive Medicine",
    "Spectroscopy",
    "Museology",
    "Gender Studies",
    "Nutrition and Dietetics",
    "Ecological Modeling",
    "General Agricultural and Biological Sciences",
    "Cancer Research",
    "Statistical and Nonlinear Physics",
    "Cellular and Molecular Neuroscience",
    "Environmental Chemistry",
    "Management Information Systems",
    "Marketing",
    "Electronic, Optical and Magnetic Materials",
    "General Economics, Econometrics and Finance",
    "Music",
    "Statistics and Probability",
    "Classics",
    "Geography, Planning and Development",
    "Communication",
    "Inorganic Chemistry",
    "Safety Research",
    "Polymers and Plastics",
    "Applied Mathematics",
    "Physical and Theoretical Chemistry",
    "Complementary and alternative medicine",
    "Hematology",
    "Agronomy and Crop Science",
    "Management of Technology and Innovation",
    "Obstetrics and Gynecology",
    "Geometry and Topology",
    "Computer Science Applications",
    "Automotive Engineering",
    "Pollution",
    "Radiation",
    "Soil Science",
    "Ophthalmology",
    "Dermatology",
    "Health",
    "Safety, Risk, Reliability and Quality",
    "Transportation",
    "Media Technology",
    "Animal Science and Zoology",
    "Neurology",
    "Genetics",
    "Mathematical Physics",
    "Emergency Medicine",
    "Statistics, Probability and Uncertainty",
    "Industrial and Manufacturing Engineering",
    "General Social Sciences",
    "Signal Processing",
    "Pharmacology",
    "Conservation",
    "Speech and Hearing",
    "Aquatic Science",
    "Orthopedics and Sports Medicine",
    "Emergency Medical Services",
    "Biotechnology",
    "Nephrology",
    "Analytical Chemistry",
    "Hepatology",
    "Geology",
    "Linguistics and Language",
    "Physical Therapy, Sports Therapy and Rehabilitation",
    "Health Information Management",
    "Parasitology",
    "Geochemistry and Petrology",
    "Earth-Surface Processes",
    "Hardware and Architecture",
    "Small Animals",
    "Development",
    "Forestry",
    "Microbiology",
    "Rehabilitation",
    "Surfaces, Coatings and Films",
    "Biochemistry",
    "Instrumentation",
    "Oral Surgery",
    "Endocrinology",
    "Occupational Therapy",
    "Pharmaceutical Science",
    "Human-Computer Interaction",
    "Public Administration",
    "Urology",
    "Clinical Biochemistry",
    "Pharmacy",
    "Gastroenterology",
    "General Materials Science",
    "Anesthesiology and Pain Medicine",
    "Periodontics",
    "Library and Information Sciences",
    "Architecture",
    "Immunology and Allergy",
    "Applied Psychology",
    "Modeling and Simulation",
    "Fluid Flow and Transfer Processes",
    "Radiological and Ultrasound Technology",
    "Algebra and Number Theory",
    "Catalysis",
    "Ceramics and Composites",
    "Otorhinolaryngology",
    "Computer Graphics and Computer-Aided Design",
    "Endocrine and Autonomic Systems",
    "Physiology",
    "Software",
    "Critical Care and Intensive Care Medicine",
    "Numerical Analysis",
    "Virology",
    "General Arts and Humanities",
    "Orthodontics",
    "Theoretical Computer Science",
    "Molecular Medicine",
    "Geriatrics and Gerontology",
    "Anatomy",
    "Sensory Systems",
    "Biochemistry",
    "Structural Biology",
    "Bioengineering",
    "Human Factors and Ergonomics",
    "Internal Medicine",
    "Energy Engineering and Power Technology",
    "Electrochemistry",
    "Discrete Mathematics and Combinatorics",
    "Issues, ethics and legal aspects",
    "Archeology",
    "Developmental Neuroscience",
    "General Psychology",
    "Toxicology",
    "Industrial relations",
    "Business and International Management",
    "Complementary and Manual Therapy",
    "General Energy",
    "Neuropsychology and Physiological Psychology",
    "Medical Laboratory Technology",
    "Behavioral Neuroscience",
    "Transplantation",
    "Health Informatics",
    "Space and Planetary Science",
    "Tourism, Leisure and Hospitality Management",
    "Developmental Biology",
    "Applied Microbiology and Biotechnology",
    "Life-span and Life-course Studies",
    "Process Chemistry and Technology",
    "Aging",
    "General Engineering",
    "Family Practice",
    "Metals and Alloys",
    "General Decision Sciences",
    "Biological Psychiatry",
    "Equine",
    "Microbiology",
    "Filtration and Separation",
    "General Dentistry",
    "Leadership and Management",
    "Fuel Technology",
    "Chemical Health and Safety",
    "Horticulture",
    "Medical Terminology",
    "Research and Theory",
    "Acoustics and Ultrasonics",
    "Computational Mathematics",
    "Drug Discovery",
    "Nuclear Energy and Engineering",
]

class LLMEntityExtractor:
    """Extractor de investigadores usando LLMs."""

    def __init__(
        self,
        *,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        available_topics: Optional[List[str]] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = get_llm_client(provider=llm_provider, model=llm_model)
        self.available_topics = (
            available_topics if available_topics is not None else OPENALEX_TOPICS
        )

    def extract_researchers_from_chunk(self, chunk_text: str, chunk_id: str) -> LLMExtractionResult:
        """Extraer investigadores de un chunk."""
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en análisis de documentos académicos.",
            },
            {"role": "user", "content": self._build_prompt(chunk_text)},
        ]

        try:
            response = self.llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return parse_researcher_response(response, chunk_id, chunk_text)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                topics=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def extract_topics_from_chunk(self, chunk_text: str, chunk_id: str) -> LLMExtractionResult:
        """Extraer tópicos de un chunk."""
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en análisis de documentos académicos.",
            },
            {"role": "user", "content": self._build_topic_prompt(chunk_text)},
        ]

        try:
            response = self.llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return parse_topic_response(response, self.available_topics, chunk_id, chunk_text)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                topics=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def _build_prompt(self, chunk_text: str) -> str:
        """Construir prompt para extracción de investigadores."""
        return f"""Extract researchers who PARTICIPATED in this research project from the text below.

RULES:
- Extract ONLY project team members (NOT cited authors from references)
- One person per entry
- Evidence must be actual text containing the person's name

TEXT:
{chunk_text}

IMPORTANT: You MUST wrap your response with <JSON> and </JSON> tags.

OUTPUT FORMAT:
<JSON>
{{
  "researchers": [
    {{"name": "Person Name", "evidence": "actual text mentioning Person Name"}}
  ]
}}
</JSON>

If no researchers found:
<JSON>{{"researchers": []}}</JSON>
"""

    def _build_topic_prompt(self, chunk_text: str) -> str:
        """Construir prompt para extracción de tópicos."""
        if self.available_topics:
            topics_list = "\n".join(f"- {topic}" for topic in self.available_topics)

            return f"""Match research topics from Spanish text to this English topic list.

ALLOWED TOPICS:
{topics_list}

TEXT (Spanish):
{chunk_text}

RULES:
- ONLY use topics from the list above
- DO NOT invent new topics
- Match by research field, not literal translation

IMPORTANT: You MUST wrap your response with <JSON> and </JSON> tags.

OUTPUT FORMAT:
<JSON>
{{
  "topics": [
    {{"topic": "Topic from list", "evidence": "Spanish text fragment"}}
  ]
}}
</JSON>

If no match:
<JSON>{{"topics": []}}</JSON>"""
        else:
            return f"""You are an information extraction system.

    Task:
    Identify the main research topics mentioned in the text.

    TEXT:
    {chunk_text}

    Rules:
    - Be concise and specific
    - Use simple, clear topic names
    - Extract only topics that are clearly mentioned in the text

    OUTPUT FORMAT - CRITICAL:
    - Wrap your JSON response in <JSON> tags
    - Format: <JSON>{{...}}</JSON>
    - Do NOT add any text before <JSON> or after </JSON>
    - Do NOT add explanations or comments
    - The JSON inside must be valid

    Exact format to follow:
    <JSON>
    {{
    "topics": [
        {{
        "topic": "Topic name",
        "evidence": "Exact text fragment mentioning it"
        }}
    ]
    }}
    </JSON>

    If no topics are identified:
    <JSON>{{"topics": []}}</JSON>
    
    Remember: Always use <JSON> tags around your response.
    """

    def extract_researchers_from_chunks(
        self, chunks: List[Dict[str, Any]], max_chunks: Optional[int] = None
    ) -> LLMExtractionResult:
        """Extraer investigadores desde lista de chunks."""
        all_researchers = []
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            if not chunk_text.strip():
                continue

            logger.debug(f"  Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_researchers_from_chunk(chunk_text, chunk_id)
            all_researchers.extend(result.researchers)
            all_errors.extend(result.errors)
            if result.researchers:
                logger.debug(
                    f"    → Encontrados: {', '.join([r.name for r in result.researchers])}"
                )

        return LLMExtractionResult(researchers=all_researchers, topics=[], errors=all_errors)

    def extract_topics_from_chunks(
        self, chunks: List[Dict[str, Any]], max_chunks: Optional[int] = None
    ) -> LLMExtractionResult:
        """Extraer tópicos desde lista de chunks."""
        all_topics = []
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            if not chunk_text.strip():
                continue

            logger.debug(f"  Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_topics_from_chunk(chunk_text, chunk_id)
            all_topics.extend(result.topics)
            all_errors.extend(result.errors)
            if result.topics:
                logger.debug(f"    → Encontrados: {', '.join([t.topic for t in result.topics])}")

        return LLMExtractionResult(researchers=[], topics=all_topics, errors=all_errors)


def create_entities_and_relationships_from_llm_extraction(
    llm_result: LLMExtractionResult,
    project_id: str,
    existing_researcher_ids: Optional[set[str]] = None,
) -> tuple[List[Entity], List[Relationship]]:
    """Crear entidades y relaciones con deduplicación."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    existing_ids = existing_researcher_ids or set()
    researchers_by_name: dict[str, str] = {}

    for mention in llm_result.researchers:
        name_normalized = mention.name.lower().strip()
        researcher_id = name_normalized.replace(" ", "_").replace(".", "").replace(",", "")

        if researcher_id in existing_ids:
            relationships.append(EXTRAIDO_DE(mention.chunk_id, researcher_id))
            continue

        if name_normalized in researchers_by_name:
            relationships.append(
                EXTRAIDO_DE(mention.chunk_id, researchers_by_name[name_normalized])
            )
            continue

        entities.append(
            Investigador(id=researcher_id, value={"name": mention.name, "source": "llm"})
        )
        relationships.append(PARTICIPO_EN(researcher_id, project_id))
        relationships.append(
            EXTRAIDO_DE(
                mention.chunk_id, researcher_id, properties={"evidence_text": mention.evidence}
            )
        )

        researchers_by_name[name_normalized] = researcher_id
        existing_ids.add(researcher_id)

    return entities, relationships


def create_topics_from_llm_extraction(
    llm_result: LLMExtractionResult,
    existing_topic_ids: Optional[set[str]] = None,
) -> tuple[List[Entity], List[Relationship]]:
    """Crear entidades y relaciones de tópicos con deduplicación global.

    Solo crea entidades Topico y relaciones EXTRAIDO_DE desde chunks.
    Las relaciones TIENE_TOPICO proyecto->topico se crean después por agregación.
    """
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    existing_ids = existing_topic_ids or set()
    topics_by_name: dict[str, str] = {}

    for mention in llm_result.topics:
        topic_normalized = mention.topic.lower().strip()
        topic_id = topic_normalized.replace(" ", "_").replace(",", "").replace("/", "_")

        # El tópico ya existe globalmente
        if topic_id in existing_ids:
            relationships.append(
                EXTRAIDO_DE(
                    mention.chunk_id, topic_id, properties={"evidence_text": mention.evidence}
                )
            )
            continue

        # El tópico ya fue creado en esta misma llamada
        if topic_normalized in topics_by_name:
            existing_topic_id = topics_by_name[topic_normalized]
            relationships.append(
                EXTRAIDO_DE(
                    mention.chunk_id,
                    existing_topic_id,
                    properties={"evidence_text": mention.evidence},
                )
            )
            continue

        entities.append(Topico(id=topic_id, value=mention.topic))
        relationships.append(
            EXTRAIDO_DE(mention.chunk_id, topic_id, properties={"evidence_text": mention.evidence})
        )

        topics_by_name[topic_normalized] = topic_id
        existing_ids.add(topic_id)

    return entities, relationships
