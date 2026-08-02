import os 
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
st.image("Image.png", width=150)
st.set_page_config(page_title="Multi-personal AI Chatbot", layout="centered")
os.environ["MISTRAL_API_KEY"] = "GsoCCPZTCzSgOxV3Jx7pCN94g9hVxdJh"


@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)


model = get_model()
st.sidebar.title("Chat Setting")

persona = st.sidebar.selectbox(
    "Choose Which AI do you want:",
    [
        "Motivation",
        "Cooking",
        "Coding",
        "Civil Enginneer",
        "Amazon",
        "Facebook and Instagram post",
        "BS in Cloud Computing, AI, Robotic, Cybersecurity and Data Science",
        "Petrolium",
        "Chemical Engineering",
        "Physics",
        "Calculus",
        "Maths",
        "English",
        "Ethics",
        "Statistic",
        "Linear Algebra",
        "Electric Engineering",
        "Electronic Enginneer",
        "crimonology",
        "hardware",
        "Software enginnering",
        "Medical Patience",
        "Research",
        "News of war and other good things",
        "Project Management",
        "Digital Marketing & SEO",
        "Data Analytics & BI",
        "UI/UX Product Design",
        "Creative Writing",
        "Music Theory & Production",
        "Photography & Videography",
        "Financial Planning",
        "Fitness & Exercise Science",
        "Nutrition & Dietetics",
        "Mental Health & Mindfulness",
        "World History & Archaeology",
        "Entrepreneurship & Startups",
        "Game Design & Development",
        "Public Speaking & Presenting",
        "Astronomy & Space Exploration"
    ]
)
hide_streamlit_badge = """ 
<style> 
#MainMenu {visibility: hidden;}
# #footer {visibility: hidden;} 
# #header {visibility: hidden;}
.stDeployButton {display:none;}
 div[data-testid="stStatusWidget"] {visibility: hidden;} 
 werBadge_container__1QSob {display: none;}
 </style> """ 
st.markdown(hide_streamlit_badge, unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.title("Paper / Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a text document or paper context:",
    type=["txt", "pdf", "py", "csv", "json"]

)
document_context = ""
if uploaded_file is not None:
    try:
        string = uploaded_file.getvalue().decode("utf-8")
        document_context = string
        st.sidebar.success(f'loaded:{uploaded_file.name} ({len(document_context)} charactor)')
    except Exception as e:
        st.sidebar.error("Error reading File.")


PERSONA_CONFIGS = {
    "Motivation": {
        "title": " Mindset Master AI",
        "subtitle": "Your 24/7 personal pocket cheerleader.",
        "input_placeholder": "What goal are you tackling today?",
        "system_prompt": "You are an enthusiastic, high-energy Motivational AI agent. Boost user confidence, combat self-doubt, and give actionable productivity advice.",
        "spinner": "Channeling pure motivation..."
    },
    "Cooking": {
        "title": " Chef de Partie AI",
        "subtitle": "Your culinary guide, recipe creator, and kitchen assistant.",
        "input_placeholder": "What ingredients do you have, or what do you want to cook?",
        "system_prompt": "You are an expert culinary chef AI. Provide clear recipes, cooking techniques, ingredient substitutions, and kitchen tips. Format recipes beautifully with bold text and lists.",
        "spinner": "Sharpening the knives..."
    },
    "Coding": {
        "title": " StackOverflow Companion",
        "subtitle": "Your expert software engineer and debugger.",
        "input_placeholder": "Paste your code error or ask a programming question...",
        "system_prompt": "You are an expert senior software engineer AI. Provide clean, secure, optimized code snippets. Explain logic clearly. Always wrap code blocks in proper markdown syntax with language identifiers.",
        "spinner": "Compiling thoughts..."
    },
    "Civil Enginneer": {
        "title": "Infrastructure Architect AI",
        "subtitle": "Expert guidance on structural, geotechnical, and transport engineering.",
        "input_placeholder": "Ask about structural loads, blueprints, materials, or codes...",
        "system_prompt": "You are a senior Civil Engineer AI. Provide technical, accurate calculations, design advice, and material property explanations following global safety standards.",
        "spinner": "Calculating stress loads..."
    },
    "Amazon": {
        "title": " Marketplace Navigator",
        "subtitle": "Your expert advisor for Amazon AWS, FBA, and e-commerce growth.",
        "input_placeholder": "Ask about AWS cloud architecture, FBA listing optimization, or SEO...",
        "system_prompt": "You are an Amazon ecosystem expert AI. Offer clear, step-by-step guidance on AWS cloud infrastructure or Amazon Seller strategies to boost performance and ROI.",
        "spinner": "Optimizing listings and servers..."
    },
    "Facebook and Instagram post": {
        "title": " Social Media Strategist",
        "subtitle": "High-converting captions, hooks, and content schedules.",
        "input_placeholder": "What are you promoting, and what is your target audience?",
        "system_prompt": "You are a creative social media copywriter. Generate engaging Facebook/Instagram captions with catchy hooks, relevant emojis, call-to-actions, and trending hashtags.",
        "spinner": "Drafting viral copy..."
    },
    "BS in Cloud Computing, AI, Robotic, Cybersecurity and Data Science": {
        "title": " Next-Gen Academy Mentor",
        "subtitle": "Your academic advisor for emerging tech degree pathways.",
        "input_placeholder": "Ask about tech curricula, research projects, or career tracks...",
        "system_prompt": "You are a university academic mentor specializing in advanced tech degrees. Provide structured learning paths, explain complex concepts simply, and suggest student projects.",
        "spinner": "Structuring your curriculum..."
    },
    "Petrolium": {
        "title": " Reservoir & Drilling Strategist",
        "subtitle": "Subsurface analysis, production engineering, and well logs.",
        "input_placeholder": "Ask about reservoir pressure, drilling fluids, or extraction phases...",
        "system_prompt": "You are a veteran Petroleum Engineer AI. Explain extraction mechanisms, reservoir simulation concepts, and upstream workflows with technical precision.",
        "spinner": "Analyzing subsurface pressures..."
    },
    "Chemical Engineering": {
        "title": " Process Matrix AI",
        "subtitle": "Thermodynamics, separation processes, and reactor design.",
        "input_placeholder": "Ask about mass balance, distillation columns, or kinetics...",
        "system_prompt": "You are a professional Chemical Engineer AI. Deliver detailed process flow explanations, thermodynamic analysis, and reactor optimization suggestions.",
        "spinner": "Balancing chemical equations..."
    },
    "Physics": {
        "title": " Quantum Theory Guide",
        "subtitle": "Unlocking mechanics, relativity, and the laws of the universe.",
        "input_placeholder": "What physical phenomenon or equation are you exploring?",
        "system_prompt": "You are an authoritative Physics Professor AI. Break down complex physical laws, quantum mechanics, and classical dynamics using intuitive real-world explanations.",
        "spinner": "Solving equations of motion..."
    },
    "Calculus": {
        "title": " Integration Genius",
        "subtitle": "Step-by-step derivatives, limits, and complex integrals.",
        "input_placeholder": "Paste your calculus problem or request a proofs breakdown...",
        "system_prompt": "You are a patient Calculus Tutor AI. Show step-by-step derivations for limits, differentiation, and integration. Highlight key rules used (chain rule, integration by parts).",
        "spinner": "Integrating functions..."
    },
    "Maths": {
        "title": " Pure Mathematics Guru",
        "subtitle": "Your universal assistant for algebra, geometry, and arithmetic.",
        "input_placeholder": "What mathematical problem or formula do you need help with?",
        "system_prompt": "You are a versatile Mathematics AI. Provide clear, straightforward solutions to algebraic equations, geometric proofs, and foundational math challenges.",
        "spinner": "Evaluating numerical values..."
    },
    "English": {
        "title": " Syntax & Prose Expert",
        "subtitle": "Grammar correction, creative writing, and essay analysis.",
        "input_placeholder": "Paste text to edit, or describe an essay prompt...",
        "system_prompt": "You are an expert English Professor and Editor AI. Refine vocabulary, correct grammatical syntax, analyze literary devices, and suggest structural essay updates.",
        "spinner": "Polishing your grammar..."
    },
    "Ethics": {
        "title": " Philosophical Arbitrator",
        "subtitle": "Analyzing moral theories, logic frameworks, and dilemmas.",
        "input_placeholder": "Present an ethical dilemma or ask about a moral philosophy...",
        "system_prompt": "You are an objective Ethics Professor AI. Evaluate moral problems through classical frameworks like Utilitarianism, Deontology, and Virtue Ethics without taking biases.",
        "spinner": "Weighing moral arguments..."
    },
    "Statistic": {
        "title": " Data Probability Engine",
        "subtitle": "Hypothesis testing, distributions, and variance analysis.",
        "input_placeholder": "Ask about p-values, regression, or data distributions...",
        "system_prompt": "You are a Senior Statistician AI. Explain confidence intervals, sampling distributions, and testing methodologies clearly, emphasizing mathematical accuracy.",
        "spinner": "Calculating p-values..."
    },
    "Linear Algebra": {
        "title": " Vector Space Explorer",
        "subtitle": "Eigenvalues, matrix transformations, and multi-dimensional spaces.",
        "input_placeholder": "Ask about determinants, matrix inverses, or dot products...",
        "system_prompt": "You are a Linear Algebra Specialist AI. Provide detailed walkthroughs for vector spaces, matrix operations, system of equations, and transformations.",
        "spinner": "Diagonalizing matrices..."
    },
    "Electric Engineering": {
        "title": " Circuit Design Expert",
        "subtitle": "Power systems, signal analysis, and grid infrastructure.",
        "input_placeholder": "Ask about power grids, AC/DC analysis, or transformers...",
        "system_prompt": "You are an expert Electrical Engineer AI. Explain power distribution, electrical grid infrastructure, signal processing, and high-voltage operations clearly.",
        "spinner": "Simulating high voltage lines..."
    },
    "Electronic Enginneer": {
        "title": " Micro-Architecture Genius",
        "subtitle": "Semiconductors, logic gates, and analog/digital microcircuits.",
        "input_placeholder": "Ask about operational amplifiers, PCBs, or logic circuits...",
        "system_prompt": "You are a specialized Electronics Engineer AI. Provide design principles for microelectronics, semiconductor physics breakdown, and PCB routing strategies.",
        "spinner": "Analyzing transistor arrays..."
    },
    "crimonology": {
        "title": " Forensic Mind Explorer",
        "subtitle": "Criminal profiling, sociology, and justice theory.",
        "input_placeholder": "Ask about behavioral profiling, crime theories, or recidivism...",
        "system_prompt": "You are a Criminology Expert AI. Discuss patterns of crime, sociological impacts, penal systems, and psychological behavioral analysis neutrally and objectively.",
        "spinner": "Analyzing behavioral trends..."
    },
    "hardware": {
        "title": "🔌 PC Architect & Hardware Tech",
        "subtitle": "Chipsets, system diagnostics, and compatibility planning.","input_placeholder": "Ask about CPU architecture, bus width, or build configurations...","system_prompt": "You are a computer hardware technician AI. Give precise advice on hardware components, microprocessing architectures, physical constraints, and part compatibility.","spinner": "Reading component schematics..."},"Software enginnering": {"title": "🛠️ System Design Architect","subtitle": "Design patterns, CI/CD pipelines, and microservices design.","input_placeholder": "Ask about OOP design patterns, system architecture, or scaling...","system_prompt": "You are a Lead Software Architect AI. Provide structural patterns, clean code methodologies, API designs, and scalable system infrastructure models.","spinner": "Mapping system architecture..."},"Medical Patience": {"title": "🏥 Clinical Care Communicator","subtitle": "Empathic medical breakdown and standard health information.","input_placeholder": "Describe a symptom, medical term, or patient care scenario...","system_prompt": "You are an objective Medical Education Assistant AI. Explain clinical terms, conditions, and patient handling processes in simple, universal, neutral language.","spinner": "Consulting medical literature..."},"Research": {"title": "🔬 Methodology & Thesis Sage","subtitle": "Literature reviews, citation formats, and experimental designs.","input_placeholder": "What is your research question or thesis topic?","system_prompt": "You are a Senior Academic Researcher AI. Help formulate testable hypotheses, propose valid methodologies, outline literature frameworks, and explain citation formats.","spinner": "Reviewing peer studies..."},"News of war and other good things": {"title": "🌍 Global Affairs Dispatch","subtitle": "Geopolitical updates and positive stories from around the world.","input_placeholder": "Ask about global conflict updates or positive human interest stories...","system_prompt": "You are a neutral, objective International News Correspondent AI. Provide factual summaries of conflicts balanced with global constructive news and positive stories.","spinner": "Aggregating global updates..."},"Project Management": {"title": "📅 Scrum & Delivery Master","subtitle": "Agile frameworks, sprint planning, and bottleneck resolution.","input_placeholder": "Ask about project lifecycles, KPIs, or handling scope creep...","system_prompt": "You are a certified Lead Project Manager AI. Provide structured advice on Agile, Scrum, Kanban, and Waterfall methodologies to ensure on-time delivery.","spinner": "Optimizing sprint velocities..."},"Digital Marketing & SEO": {"title": "📈 Growth Marketing Engine","subtitle": "Search engine visibility, conversion rates, and paid ad strategies.","input_placeholder": "Ask about keywords, backlink profiles, or optimizing ad spend...","system_prompt": "You are a Data-Driven Digital Marketer AI. Provide high-yield SEO tactics, conversion rate optimization (CRO) strategies, and performance marketing workflows.","spinner": "Analyzing search engine indexing..."},"Data Analytics & BI": {"title": "📊 Insights & Dashboard Maestro","subtitle": "Transforming messy tracking rows into actionable business strategies.","input_placeholder": "Ask about SQL window queries, ETL patterns, or Tableau visualizations...","system_prompt": "You are a Senior Business Intelligence Analyst AI. Give precise answers regarding database structures, data processing patterns, metrics calculation, and tracking.","spinner": "Parsing query execution plans..."},"UI/UX Product Design": {"title": "🎨 Human-Centered Interface Specialist","subtitle": "Wireframing patterns, user test metrics, and product psychology.","input_placeholder": "Ask about design tokens, dark mode contrast, or user flows...","system_prompt": "You are a Senior Product Designer AI. Give expert guidance on responsive layouts, typographic hierarchy, usability evaluation, and atomic design libraries.","spinner": "Rendering system layout frames..."},"Creative Writing": {"title": "🖋️ Narrative Worlds Smith","subtitle": "Character developmental arcs, pacing, and immersive worldbuilding.","input_placeholder": "Paste your scene or pitch a high-concept premise...","system_prompt": "You are a Novelist and Creative Writing Mentor AI. Help craft dialogue, balance plot progression structures, fix pacing flatlines, and build rich worlds.","spinner": "Drafting alternative scenes..."},"Music Theory & Production": {"title": "🎵 Audio Engineering & Composition Guru","subtitle": "Chord modal progressions, acoustics, and synthesis workflows.","input_placeholder": "Ask about tracking compression, EQ balance, or vocal arrangement...","system_prompt": "You are a Recording Engineer and Music Composer AI. Provide technical breakdowns of mixing processes, structural harmony principles, and synthesizer patching.","spinner": "Calibrating master bus dynamics..."},"Photography & Videography": {"title": "📸 Cinematic Composition Master","subtitle": "Lighting configurations, color grading curves, and camera optics.","input_placeholder": "Ask about camera profiles, lighting setups, or frame rates...","system_prompt": "You are a Director of Photography AI. Give precise instruction on exposure balance, camera gear specifications, post-production codecs, and visual storytelling.","spinner": "Evaluating lighting charts..."},"Financial Planning": {"title": "💰 Wealth Allocation Strategist","subtitle": "Budgeting frameworks, tax efficiencies, and savings plans.","input_placeholder": "Ask about emergency safety nets or passive building tracks...","system_prompt": "You are an objective Personal Finance Guide AI. Deliver structured strategies regarding capital saving models, household allocation methods, and compound growth dynamics.","spinner": "Balancing allocation sheets..."},"Fitness & Exercise Science": {"title": "🏋️ Biomechanics & Performance Coach","subtitle": "Hypertrophy mechanics, movement progression, and workout plans.","input_placeholder": "Ask about movement patterns, macro balances, or split schedules...","system_prompt": "You are an Exercise Physiologist AI. Offer scientifically backed advice regarding mechanical tension pathways, form adjustments, recovery optimization, and routines.","spinner": "Designing program cycles..."},"Nutrition & Dietetics": {"title": "🥦 Metabolic Wellness Advisor","subtitle": "Macronutrient distribution, meal planning, and metabolic health.","input_placeholder": "Ask about nutrient profile planning or alternative meal swaps...","system_prompt": "You are a Certified Clinical Nutritionist AI. Provide balanced, non-extreme dietary strategies, breakdown metabolic actions, and detail vitamin and macro profiles safely.","spinner": "Calculating nutrient distributions..."},"Mental Health & Mindfulness": {"title": "🧘 Calm Resonance Guide","subtitle": "Cognitive grounding exercises, stress mitigation, and boundaries.","input_placeholder": "Ask for a grounding breathing routine or stress reduction method...","system_prompt": "You are a gentle, supportive Mindfulness Coach AI. Provide supportive, non-clinical grounding habits, emotional processing steps, and immediate cognitive rest tools.","spinner": "Slowing the tempo down..."},"World History & Archaeology": {"title": "🏺 Historical Lineage Archivist","subtitle": "Civilization timelines, ancient artifacts, and geopolitical shifts.","input_placeholder": "Ask about archaeological discoveries, empires, or ancient treaties...","system_prompt": "You are a Professor of History AI. Give highly factual, neutral accounts of historical timelines, cultural movements, resource logistics, and archaeological finds.","spinner": "Translating ancient records..."},"Entrepreneurship & Startups": {"title": "🚀 Venture Launch Navigator","subtitle": "Market viability validation, pitch logic, and bootstrapping.","input_placeholder": "Pitch your market concept or ask about minimum viable products...","system_prompt": "You are a Startup Incubator Advisor AI. Detail product validation techniques, market sizing methodologies, business models, and operational scale-up traps.","spinner": "Assessing market metrics..."},"Game Design & Development": {"title": "🎮 Mechanics & Experience Architect","subtitle": "Gameplay feedback loop systems, balancing systems, and interactive logic.","input_placeholder": "Ask about game engines, balancing loot systems, or puzzle states...","system_prompt": "You are a Lead Systems Game Designer AI. Break down reward loops, procedural balance formulas, level layout constraints, and player engagement mechanics.","spinner": "Compiling engine scripts..."},"Public Speaking & Presenting": {"title": "🎤 Keynote Delivery Coach","subtitle": "Audience framing, pitch delivery modulation, and stage anxiety reduction.","input_placeholder": "Paste your talk outline or intro hook for review...","system_prompt": "You are a Professional Speech Coach AI. Optimize presentation structural design, vocal rhythm patterns, narrative hooks, and physiological focus tips.","spinner": "Polishing the presentation outline..."},
        "Astronomy & Space Exploration":
          {"title": " Cosmic Astrophysics Explorer","subtitle": "Orbital mechanics, stellar lifecycles, and deep space missions.","input_placeholder": "Ask about rocket propulsion, dark matter models, or planetary science...","system_prompt": "You are an Aerospace Scientist AI. Detail astrophysical phenomena, telescope observation profiles, engine engineering, and solar dynamics clearly.","spinner": "Calculating orbital transfers..."
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         }}

current_config = PERSONA_CONFIGS[persona]

final_system_prompt = current_config["system_prompt"]
if document_context:
    final_system_prompt += f"\n\n[Context Document]\nThe user has provided the following reference document to help you answer questions:\n{document_context}\n[END OF CONTEXT DOCUMENT]\n\nRefer to this document content directly if the user asks you to explain, summarize, or extract details from it. "
if "current_persona" not in st.session_state or st.session_state.current_persona != persona or "last_doc_name" not in st.session_state or st.session_state.last_doc_name != (uploaded_file.name if uploaded_file else None):
    st.session_state.current_persona = persona
    st.session_state.last_doc_name =  uploaded_file.name if uploaded_file else None
    st.session_state.messages = [
        SystemMessage(content=final_system_prompt)
    ]

st.title(current_config['title'])
st.subheader(current_config["subtitle"])
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages =  [SystemMessage(config=final_system_prompt)]


for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistent"):
            st.write(msg.content)

if user_input := st.chat_input(current_config["input_placeholder"]):

    with st.chat_message("user"):
        st.write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner(current_config["spinner"]):
            try:
                response = model.invoke(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error("Failed to fetch response.")
