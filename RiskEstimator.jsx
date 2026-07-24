import React, { useState, useMemo } from "react";

/* ---------------------------------------------------------------------
   Trained model, exported verbatim from model/full_model_export.json
   (ColumnTransformer[median-impute+scale numeric, mode-impute+one-hot
   categorical] + LogisticRegression, fit on the real METABRIC cohort).
   Target: disease-specific mortality (Died of Disease vs Living).
--------------------------------------------------------------------- */
const M = {
  intercept: -0.007676994696835693,
  coef: [0.5398918554553448,0.2414828800130889,0.053728937228011024,0.015453134846074714,0.3708334916922685,0.27726221633004106,0.10784400713359371,0.1384174901425587,-0.13570525242329803,0.020769192504312754,-0.01805695478505095,0.001105641650112262,0.0016065960691478576,0.21109170771918287,-0.20837946999992202,0.13178363059322648,-0.12907139287396702,-0.06490018709174222,0.06761242481100344,-0.27599350390867916,0.27870574162794004,0.07893602200527756,0.12171095338734743,-0.34060602440921967,0.03907057662100799,0.48742677750116997,0.0757293668822869,-0.45955543426860945,-0.13317388268142605,-0.16567692694255207,0.11385761554791923,0.18770543179532134,-0.020509117904351046,-0.5502583219215033,0.4116701352318136,-0.6703538265548548,-0.2672675988348117,-0.17569214605033515,1.014892998983216,0.40637725527808294,-0.089859080772738,-0.19099811301051992,0.13471005327526284,0.37580501860727106,0.18684593255982238,0.010007720208759818,-0.5699464336565931],
  num_cols: ["Age at Diagnosis","Tumor Size","Tumor Stage","Neoplasm Histologic Grade","Lymph nodes examined positive","Nottingham prognostic index","Mutation Count"],
  num_mean: [58.16563848920863,26.55123201438849,1.8012589928057554,2.471223021582734,2.2068345323741005,4.114739154676259,5.485611510791367],
  num_scale: [12.783408576906632,16.83208006324595,0.5792253551593,0.6354904755430862,4.257497413319124,1.1587997302184365,3.3285026155954536],
  cat_feature_names: ["ER Status_Negative","ER Status_Positive","PR Status_Negative","PR Status_Positive","Chemotherapy_No","Chemotherapy_Yes","Hormone Therapy_No","Hormone Therapy_Yes","Radio Therapy_No","Radio Therapy_Yes","Type of Breast Surgery_Breast Conserving","Type of Breast Surgery_Mastectomy","Inferred Menopausal State_Post","Inferred Menopausal State_Pre","Pam50 + Claudin-low subtype_Basal","Pam50 + Claudin-low subtype_Her2","Pam50 + Claudin-low subtype_LumA","Pam50 + Claudin-low subtype_LumB","Pam50 + Claudin-low subtype_NC","Pam50 + Claudin-low subtype_Normal","Pam50 + Claudin-low subtype_claudin-low","HER2 status measured by SNP6_Gain","HER2 status measured by SNP6_Loss","HER2 status measured by SNP6_Neutral","HER2 status measured by SNP6_Undef","Integrative Cluster_1","Integrative Cluster_10","Integrative Cluster_2","Integrative Cluster_3","Integrative Cluster_4ER+","Integrative Cluster_4ER-","Integrative Cluster_5","Integrative Cluster_6","Integrative Cluster_7","Integrative Cluster_8","Integrative Cluster_9","3-Gene classifier subtype_ER+/HER2- High Prolif","3-Gene classifier subtype_ER+/HER2- Low Prolif","3-Gene classifier subtype_ER-/HER2-","3-Gene classifier subtype_HER2+"],
};

const GENOMIC_NUM = new Set(["Mutation Count"]);
const GENOMIC_CAT_PREFIXES = [
  "Pam50 + Claudin-low subtype", "HER2 status measured by SNP6",
  "Integrative Cluster", "3-Gene classifier subtype",
];
const isGenomicCat = (name) => GENOMIC_CAT_PREFIXES.some((p) => name.startsWith(p + "_"));

const CAT_OPTIONS = {
  "ER Status": ["Negative", "Positive"],
  "PR Status": ["Negative", "Positive"],
  "Chemotherapy": ["No", "Yes"],
  "Hormone Therapy": ["No", "Yes"],
  "Radio Therapy": ["No", "Yes"],
  "Type of Breast Surgery": ["Breast Conserving", "Mastectomy"],
  "Inferred Menopausal State": ["Pre", "Post"],
  "Pam50 + Claudin-low subtype": ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal", "NC"],
  "HER2 status measured by SNP6": ["Neutral", "Gain", "Loss", "Undef"],
  "Integrative Cluster": ["1", "2", "3", "4ER+", "4ER-", "5", "6", "7", "8", "9", "10"],
  "3-Gene classifier subtype": ["ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "ER-/HER2-", "HER2+"],
};

const PAM50_INFO = {
  LumA: "Luminal A — ER+, typically slower-growing, best prognosis",
  LumB: "Luminal B — ER+, more proliferative than Luminal A",
  Her2: "HER2-enriched — HER2-driven, historically aggressive",
  Basal: "Basal-like — largely overlaps with triple-negative",
  "claudin-low": "Claudin-low — low cell-adhesion markers, mesenchymal features",
  Normal: "Normal-like — expression pattern close to normal breast tissue",
  NC: "Not classified",
};

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function computeRisk(inputs) {
  let clinicalZ = M.intercept;
  let genomicZ = 0;

  M.num_cols.forEach((col, i) => {
    const raw = Number(inputs[col]);
    const z = (raw - M.num_mean[i]) / M.num_scale[i];
    const term = M.coef[i] * z;
    if (GENOMIC_NUM.has(col)) genomicZ += term;
    else clinicalZ += term;
  });

  const numLen = M.num_cols.length;
  Object.keys(CAT_OPTIONS).forEach((col) => {
    const chosen = inputs[col];
    const fname = `${col}_${chosen}`;
    const j = M.cat_feature_names.indexOf(fname);
    if (j === -1) return;
    const idx = numLen + j;
    const term = M.coef[idx];
    if (isGenomicCat(fname)) genomicZ += term;
    else clinicalZ += term;
  });

  return {
    fullRisk: sigmoid(clinicalZ + genomicZ),
    clinicalOnlyRisk: sigmoid(clinicalZ),
  };
}

function riskCategory(p) {
  if (p < 0.20) return { label: "Low", color: "#1f9d6f" };
  if (p < 0.40) return { label: "Moderate", color: "#d9a412" };
  if (p < 0.60) return { label: "High", color: "#e0692a" };
  return { label: "Very high", color: "#c23b3b" };
}

const FIELD_LABEL = "block text-[11px] tracking-[0.08em] uppercase text-[#5b6472] mb-1.5 font-medium";
const NUM_INPUT =
  "w-full bg-[#f4f6f5] border border-[#d8ddda] rounded-md px-3 py-2 text-[15px] font-['IBM_Plex_Mono',monospace] text-[#1c2622] focus:outline-none focus:ring-2 focus:ring-[#0f766e]/40 focus:border-[#0f766e]";

function Pills({ options, value, onChange, info }) {
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={`px-2.5 py-1 rounded-full text-[11px] font-['IBM_Plex_Mono',monospace] border transition-colors ${
              value === opt
                ? "bg-[#2dd4a8] border-[#2dd4a8] text-[#0b1512]"
                : "bg-transparent border-[#3a4d46] text-[#9fb6ae] hover:border-[#2dd4a8]"
            }`}
          >
            {opt}
          </button>
        ))}
      </div>
      {info && info[value] && (
        <div className="text-[10.5px] text-[#7c8f88] mt-1.5 leading-snug">{info[value]}</div>
      )}
    </div>
  );
}

export default function MetabricRiskEstimator() {
  const [inputs, setInputs] = useState({
    "Age at Diagnosis": 61,
    "Tumor Size": 28,
    "Tumor Stage": 2,
    "Neoplasm Histologic Grade": 3,
    "Lymph nodes examined positive": 3,
    "Nottingham prognostic index": 5.1,
    "Mutation Count": 6,
    "ER Status": "Negative",
    "PR Status": "Negative",
    "Chemotherapy": "Yes",
    "Hormone Therapy": "No",
    "Radio Therapy": "Yes",
    "Type of Breast Surgery": "Mastectomy",
    "Inferred Menopausal State": "Post",
    "Pam50 + Claudin-low subtype": "Basal",
    "HER2 status measured by SNP6": "Neutral",
    "Integrative Cluster": "10",
    "3-Gene classifier subtype": "ER-/HER2-",
  });

  const setNum = (key) => (e) => setInputs((p) => ({ ...p, [key]: Number(e.target.value) }));
  const setCat = (key) => (val) => setInputs((p) => ({ ...p, [key]: val }));

  const { fullRisk, clinicalOnlyRisk } = useMemo(() => computeRisk(inputs), [inputs]);
  const cat = riskCategory(fullRisk);
  const deltaPts = (fullRisk - clinicalOnlyRisk) * 100;

  const pct = Math.min(fullRisk, 0.85) / 0.85;
  const angle = -180 + pct * 180;
  const r = 84, cx = 110, cy = 110;
  const needleX = cx + r * Math.cos((angle * Math.PI) / 180);
  const needleY = cy + r * Math.sin((angle * Math.PI) / 180);

  return (
    <div className="w-full min-h-screen bg-[#f7f8f6] text-[#1c2622] font-['IBM_Plex_Sans',sans-serif]">
      <div className="max-w-5xl mx-auto px-6 py-10">
        <div className="mb-8 border-b border-[#dfe3e0] pb-6">
          <div className="text-[11px] tracking-[0.16em] uppercase text-[#0f766e] font-semibold mb-2 font-['IBM_Plex_Mono',monospace]">
            METABRIC cohort · n=1,483 · research demo
          </div>
          <h1 className="text-[28px] sm:text-[34px] font-semibold tracking-tight text-[#12201c]">
            Breast Cancer Outcome Estimator
          </h1>
          <p className="text-[14px] text-[#5b6472] mt-1.5 max-w-xl">
            Disease-specific mortality model trained on real clinical and
            molecular data from the METABRIC study, comparing clinical
            factors alone against clinical + genomic subtyping.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_0.85fr] gap-8">
          <div className="space-y-8">
            <section>
              <h2 className="text-[13px] font-semibold tracking-[0.06em] uppercase text-[#12201c] mb-4 flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-[#6b7280]" />
                Clinical & pathological
              </h2>
              <div className="grid grid-cols-2 gap-x-5 gap-y-4">
                <div>
                  <label className={FIELD_LABEL}>Age at diagnosis</label>
                  <input type="number" className={NUM_INPUT} value={inputs["Age at Diagnosis"]} onChange={setNum("Age at Diagnosis")} min={20} max={100} />
                </div>
                <div>
                  <label className={FIELD_LABEL}>Tumor size (mm)</label>
                  <input type="number" className={NUM_INPUT} value={inputs["Tumor Size"]} onChange={setNum("Tumor Size")} min={1} max={180} />
                </div>
                <div>
                  <label className={FIELD_LABEL}>Tumor stage</label>
                  <input type="number" className={NUM_INPUT} value={inputs["Tumor Stage"]} onChange={setNum("Tumor Stage")} min={0} max={4} />
                </div>
                <div>
                  <label className={FIELD_LABEL}>Histologic grade</label>
                  <input type="number" className={NUM_INPUT} value={inputs["Neoplasm Histologic Grade"]} onChange={setNum("Neoplasm Histologic Grade")} min={1} max={3} />
                </div>
                <div>
                  <label className={FIELD_LABEL}>Positive lymph nodes</label>
                  <input type="number" className={NUM_INPUT} value={inputs["Lymph nodes examined positive"]} onChange={setNum("Lymph nodes examined positive")} min={0} max={45} />
                </div>
                <div>
                  <label className={FIELD_LABEL}>Nottingham prognostic index</label>
                  <input type="number" step="0.1" className={NUM_INPUT} value={inputs["Nottingham prognostic index"]} onChange={setNum("Nottingham prognostic index")} min={1} max={7} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-x-5 gap-y-4 mt-4">
                {["ER Status", "PR Status", "Chemotherapy", "Hormone Therapy", "Radio Therapy", "Type of Breast Surgery", "Inferred Menopausal State"].map((col) => (
                  <div key={col}>
                    <label className={FIELD_LABEL}>{col}</label>
                    <select className={NUM_INPUT} value={inputs[col]} onChange={(e) => setCat(col)(e.target.value)}>
                      {CAT_OPTIONS[col].map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-[13px] font-semibold tracking-[0.06em] uppercase text-[#12201c] mb-4 flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-[#0f766e]" />
                Molecular / genomic
              </h2>

              <div className="mb-4">
                <div className="flex justify-between items-baseline mb-1.5">
                  <label className={FIELD_LABEL + " mb-0"}>Mutation count</label>
                  <span className="text-[13px] font-['IBM_Plex_Mono',monospace] text-[#0f766e] font-medium">
                    {inputs["Mutation Count"]}
                  </span>
                </div>
                <input type="range" min={1} max={30} value={inputs["Mutation Count"]} onChange={setNum("Mutation Count")} className="w-full accent-[#0f766e]" />
              </div>

              <div className="bg-[#12201c] rounded-lg p-4 space-y-4">
                <div>
                  <div className="text-[10px] tracking-[0.1em] uppercase text-[#8fb5ab] mb-2 font-['IBM_Plex_Mono',monospace]">
                    PAM50 + claudin-low subtype
                  </div>
                  <Pills options={CAT_OPTIONS["Pam50 + Claudin-low subtype"]} value={inputs["Pam50 + Claudin-low subtype"]} onChange={setCat("Pam50 + Claudin-low subtype")} info={PAM50_INFO} />
                </div>
                <div>
                  <div className="text-[10px] tracking-[0.1em] uppercase text-[#8fb5ab] mb-2 font-['IBM_Plex_Mono',monospace]">
                    HER2 status (SNP6 copy number)
                  </div>
                  <Pills options={CAT_OPTIONS["HER2 status measured by SNP6"]} value={inputs["HER2 status measured by SNP6"]} onChange={setCat("HER2 status measured by SNP6")} />
                </div>
                <div>
                  <div className="text-[10px] tracking-[0.1em] uppercase text-[#8fb5ab] mb-2 font-['IBM_Plex_Mono',monospace]">
                    Integrative cluster
                  </div>
                  <Pills options={CAT_OPTIONS["Integrative Cluster"]} value={inputs["Integrative Cluster"]} onChange={setCat("Integrative Cluster")} />
                </div>
                <div>
                  <div className="text-[10px] tracking-[0.1em] uppercase text-[#8fb5ab] mb-2 font-['IBM_Plex_Mono',monospace]">
                    3-gene classifier subtype
                  </div>
                  <Pills options={CAT_OPTIONS["3-Gene classifier subtype"]} value={inputs["3-Gene classifier subtype"]} onChange={setCat("3-Gene classifier subtype")} />
                </div>
              </div>
            </section>
          </div>

          <div>
            <div className="bg-white border border-[#dfe3e0] rounded-xl p-6 sticky top-6">
              <div className="text-[11px] tracking-[0.1em] uppercase text-[#5b6472] mb-4 font-['IBM_Plex_Mono',monospace]">
                Estimated disease-specific mortality
              </div>

              <div className="flex justify-center mb-2">
                <svg width="220" height="130" viewBox="0 0 220 130">
                  <path d="M 26 110 A 84 84 0 0 1 194 110" fill="none" stroke="#e6e9e7" strokeWidth="14" strokeLinecap="round" />
                  <path d="M 26 110 A 84 84 0 0 1 194 110" fill="none" stroke={cat.color} strokeWidth="14" strokeLinecap="round" strokeDasharray={`${pct * 264} 264`} />
                  <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="#12201c" strokeWidth="2.5" strokeLinecap="round" />
                  <circle cx={cx} cy={cy} r="4" fill="#12201c" />
                  <text x="110" y="95" textAnchor="middle" fontSize="28" fontWeight="600" fill="#12201c" fontFamily="IBM Plex Mono, monospace">
                    {(fullRisk * 100).toFixed(1)}%
                  </text>
                </svg>
              </div>

              <div className="text-center mb-6">
                <span className="inline-block px-3 py-1 rounded-full text-[12px] font-semibold" style={{ backgroundColor: `${cat.color}1a`, color: cat.color }}>
                  {cat.label} risk
                </span>
              </div>

              <div className="border-t border-[#eef0ee] pt-4 space-y-3">
                <div className="flex justify-between text-[13px]">
                  <span className="text-[#5b6472]">Clinical-only estimate</span>
                  <span className="font-['IBM_Plex_Mono',monospace] text-[#374150]">{(clinicalOnlyRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-[13px]">
                  <span className="text-[#5b6472]">Genomic adjustment</span>
                  <span className="font-['IBM_Plex_Mono',monospace] font-medium" style={{ color: deltaPts >= 0 ? "#c23b3b" : "#1f9d6f" }}>
                    {deltaPts >= 0 ? "+" : ""}{deltaPts.toFixed(1)} pts
                  </span>
                </div>
                <div className="flex justify-between text-[13px] pt-1 border-t border-[#eef0ee]">
                  <span className="text-[#12201c] font-medium">Full model estimate</span>
                  <span className="font-['IBM_Plex_Mono',monospace] font-semibold text-[#12201c]">{(fullRisk * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <p className="text-[11px] leading-relaxed text-[#8b93a0] mt-4 px-1">
              Trained on the public METABRIC research cohort (real, de-identified
              patient data). Estimates disease-specific mortality, not overall
              survival. This is a methods demonstration, not a validated
              clinical tool — see the README for cohort details and limitations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
