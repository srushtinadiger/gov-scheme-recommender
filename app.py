"""
╔══════════════════════════════════════════════════════════════╗
║   AI Government Scheme Recommender — Flask + Transformer    ║
║                                                              ║
║   SETUP & RUN:                                               ║
║     pip install flask sentence-transformers scikit-learn     ║
║     python app.py                                            ║
║     Open: http://localhost:5000                              ║
║                                                              ║
║   AIML: Sentence-BERT (Transformer) + Cosine Similarity     ║
╚══════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer, util
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Server busy, please try again in a moment."}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Page not found."}), 404
# ─────────────────────────────────────────────
# SCHEME DATABASE
# ─────────────────────────────────────────────
SCHEMES = [
    {"id":"S01","name":"PM Kisan Samman Nidhi","ministry":"Ministry of Agriculture","type":"Central","category":"Agriculture","benefit":"₹6,000/year direct bank transfer","eligibility_text":"farmer agricultural land owner annual income below 200000 rural farming kisan small marginal cultivator"},
    {"id":"S02","name":"PM Jan Dhan Yojana","ministry":"Ministry of Finance","type":"Central","category":"Financial","benefit":"Zero-balance account + ₹1L accident insurance","eligibility_text":"no bank account unbanked citizen poor low income adult above 18 financial inclusion"},
    {"id":"S03","name":"PM Awas Yojana (Rural)","ministry":"Ministry of Rural Development","type":"Central","category":"Housing","benefit":"₹1.2–1.3 lakh for house construction","eligibility_text":"homeless no house kutcha house rural BPL below poverty line income below 300000 poor family"},
    {"id":"S04","name":"Ayushman Bharat – PMJAY","ministry":"Ministry of Health","type":"Central","category":"Health","benefit":"₹5 lakh/year health insurance","eligibility_text":"poor family low income BPL health insurance hospital SC ST OBC daily wage laborer"},
    {"id":"S05","name":"National Scholarship Portal","ministry":"Ministry of Education","type":"Central","category":"Education","benefit":"₹500–₹20,000 scholarship/year","eligibility_text":"student school college SC ST OBC minority education scholarship income below 250000"},
    {"id":"S06","name":"PM Garib Kalyan Anna Yojana","ministry":"Ministry of Consumer Affairs","type":"Central","category":"Food Security","benefit":"5 kg free food grain/month","eligibility_text":"poor family ration card BPL low income food security daily wage laborer hunger"},
    {"id":"S07","name":"PM Mudra Yojana","ministry":"Ministry of Finance","type":"Central","category":"Employment","benefit":"₹10,000–₹10 lakh collateral-free loan","eligibility_text":"self employed small business entrepreneur shop vendor artisan micro enterprise loan startup"},
    {"id":"S08","name":"Sukanya Samriddhi Yojana","ministry":"Ministry of Finance","type":"Central","category":"Women & Child","benefit":"7.6% interest savings + tax benefit","eligibility_text":"girl child daughter below 10 years female education marriage savings parent guardian"},
    {"id":"S09","name":"MGNREGS (NREGA)","ministry":"Ministry of Rural Development","type":"Central","category":"Employment","benefit":"100 days guaranteed employment/year","eligibility_text":"rural household unemployed unskilled labor daily wage worker poverty rural adult job card"},
    {"id":"S10","name":"Atal Pension Yojana","ministry":"Ministry of Finance","type":"Central","category":"Social Security","benefit":"₹1,000–₹5,000 monthly pension after 60","eligibility_text":"unorganized sector worker age 18 to 40 no pension daily wage laborer self employed bank account retirement"},
    {"id":"S11","name":"PM Jeevan Jyoti Bima Yojana","ministry":"Ministry of Finance","type":"Central","category":"Insurance","benefit":"₹2 lakh life insurance at ₹436/year","eligibility_text":"age 18 to 50 bank account life insurance affordable low income worker family protection"},
    {"id":"S12","name":"PM Suraksha Bima Yojana","ministry":"Ministry of Finance","type":"Central","category":"Insurance","benefit":"₹2 lakh accident insurance at ₹20/year","eligibility_text":"bank account age 18 to 70 accident insurance disability laborer worker low income"},
    {"id":"S13","name":"National Social Assistance Programme","ministry":"Ministry of Rural Development","type":"Central","category":"Social Security","benefit":"₹300–₹500/month pension","eligibility_text":"elderly age above 60 widow disabled person BPL below poverty line pension old age"},
    {"id":"S14","name":"Divyangjan Scholarship","ministry":"Ministry of Social Justice","type":"Central","category":"Education","benefit":"₹500–₹2,000/month + equipment support","eligibility_text":"disability disabled person student education scholarship divyang handicapped income below 250000"},
    {"id":"S15","name":"Kisan Credit Card","ministry":"Ministry of Agriculture","type":"Central","category":"Agriculture","benefit":"Credit up to ₹3 lakh at 4% interest","eligibility_text":"farmer agricultural land owner kisan crop cultivation credit loan low interest"},
    {"id":"S16","name":"Stand-Up India","ministry":"Ministry of Finance","type":"Central","category":"Employment","benefit":"₹10 lakh–₹1 crore loan for enterprise","eligibility_text":"SC ST scheduled caste scheduled tribe woman female entrepreneur startup new business loan"},
    {"id":"S17","name":"Beti Bachao Beti Padhao","ministry":"Ministry of WCD","type":"Central","category":"Women & Child","benefit":"Girl child welfare + education support","eligibility_text":"girl child female education welfare parent daughter school student gender discrimination"},
    {"id":"S18","name":"PM SVANidhi","ministry":"Ministry of Housing","type":"Central","category":"Employment","benefit":"₹10,000–₹50,000 working capital loan","eligibility_text":"street vendor hawker roadside seller cart food stall urban livelihood loan"},
    {"id":"S19","name":"PM Vishwakarma Yojana","ministry":"Ministry of MSME","type":"Central","category":"Employment","benefit":"₹15K toolkit + ₹3L loan at 5% + training","eligibility_text":"artisan craftsperson carpenter weaver blacksmith potter cobbler mason traditional craft skill"},
    {"id":"S20","name":"National SC ST Hub","ministry":"Ministry of MSME","type":"Central","category":"Employment","benefit":"Procurement preference + business support","eligibility_text":"SC ST scheduled caste tribe entrepreneur business owner MSME small medium enterprise"},

    # ── APL (ABOVE POVERTY LINE) SCHEMES ─────────────────────────
    {"id":"APL01","name":"Atal Pension Yojana","ministry":"Ministry of Finance","type":"Central","category":"Social Security","benefit":"Guaranteed pension ₹1,000–₹5,000/month after age 60","eligibility_text":"APL above poverty line working adult 18 to 40 bank account pension retirement savings middle class"},
    {"id":"APL02","name":"PM Jeevan Jyoti Bima Yojana","ministry":"Ministry of Finance","type":"Central","category":"Insurance","benefit":"₹2 lakh life insurance at ₹436/year","eligibility_text":"APL above poverty line bank account life insurance 18 to 50 affordable middle class salaried"},
    {"id":"APL03","name":"PM Suraksha Bima Yojana","ministry":"Ministry of Finance","type":"Central","category":"Insurance","benefit":"₹2 lakh accident insurance at ₹20/year","eligibility_text":"APL above poverty line accident insurance bank account 18 to 70 low premium middle class worker"},
    {"id":"APL04","name":"Senior Citizen Savings Scheme (SCSS)","ministry":"Ministry of Finance","type":"Central","category":"Financial","benefit":"8.2% interest/year — invest up to ₹30 lakh for senior citizens","eligibility_text":"APL senior citizen 60 years above savings high interest bank deposit retirement middle class"},
    {"id":"APL05","name":"Sukanya Samriddhi Yojana","ministry":"Ministry of Finance","type":"Central","category":"Women & Child","benefit":"8.2% interest savings for girl child — education and marriage fund","eligibility_text":"APL above poverty line girl child daughter below 10 savings education marriage parent middle class"},
    {"id":"APL06","name":"National Pension System (NPS)","ministry":"Ministry of Finance","type":"Central","category":"Social Security","benefit":"Market-linked pension + tax benefit up to ₹2 lakh under 80C/80CCD","eligibility_text":"APL salaried employee NPS pension tax benefit 18 to 60 working professional middle class government private"},
    {"id":"APL07","name":"PMEGP (Employment Generation Programme)","ministry":"Ministry of MSME","type":"Central","category":"Employment","benefit":"Subsidy 15–35% on project up to ₹50 lakh for new micro enterprises","eligibility_text":"APL above poverty line entrepreneur new business manufacturing service subsidy loan 18 years educated"},
    {"id":"APL08","name":"Kisan Credit Card (APL Farmers)","ministry":"Ministry of Agriculture","type":"Central","category":"Agriculture","benefit":"Credit up to ₹3 lakh at 4% interest for APL farmers","eligibility_text":"APL above poverty line farmer agricultural land owner kisan credit crop cultivation loan"},
    {"id":"APL09","name":"PM Fasal Bima Yojana (APL Farmers)","ministry":"Ministry of Agriculture","type":"Central","category":"Insurance","benefit":"Crop insurance at low premium against natural disasters","eligibility_text":"APL above poverty line farmer crop insurance natural disaster flood drought pest disease kharif rabi"},
    {"id":"APL10","name":"Income Tax Deduction 80C / 80D","ministry":"Ministry of Finance","type":"Central","category":"Financial","benefit":"Tax deduction ₹1.5 lakh (80C) + ₹25,000 health insurance (80D) for taxpayers","eligibility_text":"APL taxpayer income tax deduction 80C 80D LIC PPF health insurance salaried professional above poverty"},
    {"id":"APL11","name":"PM Awas Yojana Urban (MIG)","ministry":"Ministry of Housing","type":"Central","category":"Housing","benefit":"Interest subsidy 3–4% on home loan for Middle Income Group first home buyers","eligibility_text":"APL above poverty line middle income group MIG home loan interest subsidy first house urban"},
    {"id":"APL12","name":"MSME Credit Guarantee Scheme","ministry":"Ministry of MSME","type":"Central","category":"Employment","benefit":"Collateral-free loan up to ₹2 crore for MSME owners through CGTMSE","eligibility_text":"APL above poverty line MSME small medium enterprise collateral free loan credit guarantee business owner"},
    {"id":"APL13","name":"Stand-Up India (APL Women/SC/ST)","ministry":"Ministry of Finance","type":"Central","category":"Employment","benefit":"Loan ₹10 lakh–₹1 crore for SC/ST and women to set up new enterprise","eligibility_text":"APL above poverty line SC ST woman female entrepreneur new enterprise bank loan 18 to 65"},
    {"id":"APL14","name":"Startup India Seed Fund","ministry":"Ministry of Commerce","type":"Central","category":"Employment","benefit":"Up to ₹20 lakh grant for startup proof-of-concept and prototype development","eligibility_text":"APL entrepreneur innovator startup young 18 to 45 seed fund grant technology product above poverty line"},
    {"id":"APL15","name":"CGHS Health Coverage (Central Govt Employees)","ministry":"Ministry of Health","type":"Central","category":"Health","benefit":"Comprehensive health coverage for central government employees and families","eligibility_text":"APL central government employee salaried CGHS health insurance family medical subsidised above poverty"},
]

# ─────────────────────────────────────────────
# TRANSFORMER MODEL (loaded once at startup)
# ─────────────────────────────────────────────
print("[AI] Loading Sentence-BERT transformer model...")
if TORCH_AVAILABLE:
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SCHEME_EMBEDDINGS = MODEL.encode(
        [s["eligibility_text"] for s in SCHEMES],
        convert_to_tensor=True, show_progress_bar=False
    )
    print(f"[AI] Model ready! {len(SCHEMES)} schemes embedded.")
else:
    MODEL = None
    np.random.seed(42)
    SCHEME_EMBEDDINGS = np.random.rand(len(SCHEMES), 384)
    print("[WARN] sentence-transformers not installed. Using mock embeddings.")


# ─────────────────────────────────────────────
# FEATURE ENGINEERING: profile → NLP text
# ─────────────────────────────────────────────
def profile_to_text(p):
    parts = []
    age, income = int(p.get("age", 30)), int(p.get("income", 100000))
    occ, cat, gender = p.get("occupation","").lower(), p.get("category",""), p.get("gender","").lower()

    parts.append("elderly senior above 60" if age >= 60 else f"young adult age {age}" if age <= 25 else f"adult age {age}")
    parts.append("very low income poor BPL below poverty" if income < 100000 else f"low income {income}" if income < 250000 else f"middle income APL above poverty line {income}")
    parts.append(f"occupation {occ}")

    kw = {"farmer":"farmer agricultural kisan land crop","daily wage":"daily wage laborer unskilled rural poor",
          "self-employed":"self employed entrepreneur business shop","student":"student education college young",
          "street vendor":"street vendor hawker roadside cart","artisan":"artisan craftsperson traditional skill",
          "fisherman":"fisherman fishing coastal marine","construction":"construction mason builder labor",
          "homemaker":"homemaker housewife domestic home","unemployed":"unemployed no job no income",
          "salaried":"salaried employee working professional middle class"}
    for k, v in kw.items():
        if k in occ: parts.append(v)

    parts.append(f"category {cat}")
    if "SC" in cat: parts.append("SC scheduled caste disadvantaged")
    if "ST" in cat: parts.append("ST scheduled tribe disadvantaged")
    if "OBC" in cat: parts.append("other backward class OBC")
    if "EWS" in cat: parts.append("economically weaker section")
    if "female" in gender or "woman" in gender: parts.append("woman female girl")
    if p.get("disability"): parts.append("disability disabled divyang handicapped")
    if p.get("bpl"):  parts.append("BPL card below poverty line ration card poor")
    if p.get("apl"):  parts.append("APL above poverty line middle class ration card not poor")
    if p.get("land"): parts.append("agricultural land owner farmer kisan")
    parts.append("bank account savings" if p.get("bank", True) else "no bank account unbanked")
    return " ".join(parts)


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(profile, threshold=0.18):
    text = profile_to_text(profile)
    if TORCH_AVAILABLE and MODEL:
        emb    = MODEL.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(emb, SCHEME_EMBEDDINGS)[0].cpu().numpy()
    else:
        np.random.seed(hash(str(sorted(profile.items()))) % 2**31)
        scores = cosine_similarity(np.random.rand(1,384), np.array(SCHEME_EMBEDDINGS))[0]

    results = []
    for i in np.argsort(scores)[::-1]:
        score = float(scores[i])
        if score < threshold: continue
        s = SCHEMES[i].copy()
        s["score"]    = round(score, 4)
        s["match"]    = round(score * 100, 1)
        s["priority"] = "HIGH" if score > 0.44 else "MEDIUM" if score > 0.30 else "LOW"
        results.append(s)
    return results, text


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    try:
        results, profile_text = recommend(data)
        high   = sum(1 for r in results if r["priority"] == "HIGH")
        medium = sum(1 for r in results if r["priority"] == "MEDIUM")
        low    = sum(1 for r in results if r["priority"] == "LOW")
        return jsonify({
            "success": True,
            "total": len(results),
            "high": high, "medium": medium, "low": low,
            "profile_text": profile_text,
            "schemes": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI Government Scheme Recommender")
    print("  Transformer: Sentence-BERT (all-MiniLM-L6-v2)")
    print("  Multi-user mode: 16 threads")
    print("  Open your browser: http://localhost:5000")
    print("="*55 + "\n")
    from waitress import serve
serve(app, host="0.0.0.0", port=7860, threads=16)
