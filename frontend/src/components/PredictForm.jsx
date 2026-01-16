import { useEffect, useMemo, useState } from "react";

export default function Predict() {
  const API = import.meta.env.VITE_API_BASE_URL;

  const [features, setFeatures] = useState([]);
  const [ranges, setRanges] = useState({});
  const [labels, setLabels] = useState({});
  const [helptext, setHelptext] = useState({});
  const [presets, setPresets] = useState({});
  const [values, setValues] = useState({});
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/model-info`);
        const info = await res.json();

        setFeatures(info.selected_features || []);
        setRanges(info.ranges || {});
        setLabels(info.labels || {});
        setHelptext(info.helptext || {});
        setPresets(info.presets || {});
        setNote(info.note || "");

        const init = {};
        const fList = info.selected_features || [];
        const benign = info.presets?.benign_typical;

        fList.forEach((f) => {
          if (benign && benign[f] != null) {
            init[f] = benign[f];
          } else {
            const min = info.ranges?.[f]?.min ?? 1;
            const max = info.ranges?.[f]?.max ?? 10;
            init[f] = Math.round((min + max) / 2);
          }
        });

        setValues(init);
      } catch (e) {
        setError("Failed to load model info.");
      }
    })();
  }, [API]);

  const canPredict = useMemo(() => {
    return features.length > 0 && features.every((f) => values[f] !== undefined && values[f] !== null);
  }, [features, values]);

  function applyPreset(key) {
    const p = presets[key];
    if (!p) return;
    setValues({ ...p });
    setResult("");
    setError("");
  }

  async function onPredict() {
    setError("");
    setResult("Predicting...");

    try {
      const payload = { features: values };

      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        setResult("");
        setError(data?.error || "Prediction failed.");
        return;
      }

      setResult(`Prediction: ${data.label}`);
    } catch (e) {
      setResult("");
      setError("Network error calling the API.");
    }
  }

  return (
    <div style={{ maxWidth: 680, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Breast Tumor Classifier</h1>
      {note && <p style={{ opacity: 0.8 }}>{note}</p>}

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", margin: "16px 0" }}>
        <button onClick={() => applyPreset("benign_typical")} disabled={!presets.benign_typical}>
          Typical Benign
        </button>
        <button onClick={() => applyPreset("malignant_typical")} disabled={!presets.malignant_typical}>
          Typical Malignant
        </button>
        <button onClick={() => applyPreset("benign_sample")} disabled={!presets.benign_sample}>
          Random Benign Sample
        </button>
        <button onClick={() => applyPreset("malignant_sample")} disabled={!presets.malignant_sample}>
          Random Malignant Sample
        </button>
      </div>

      {error && (
        <div style={{ padding: 12, borderRadius: 8, background: "#ffe5e5", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {features.map((f) => {
        const min = ranges[f]?.min ?? 1;
        const max = ranges[f]?.max ?? 10;

        return (
          <div key={f} style={{ marginBottom: 18, padding: 12, border: "1px solid #ddd", borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <div>
                <div style={{ fontWeight: 700 }}>{labels[f] || f}</div>
                <div style={{ fontSize: 13, opacity: 0.75 }}>{helptext[f] || ""}</div>
              </div>
              <div style={{ fontSize: 18, fontWeight: 800 }}>{values[f]}</div>
            </div>

            <input
              type="range"
              min={min}
              max={max}
              step="1"
              value={values[f]}
              onChange={(e) => setValues((prev) => ({ ...prev, [f]: Number(e.target.value) }))}
              style={{ width: "100%", marginTop: 10 }}
            />

            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, opacity: 0.7 }}>
              <span>{min}</span>
              <span>{max}</span>
            </div>
          </div>
        );
      })}

      <button
        onClick={onPredict}
        disabled={!canPredict}
        style={{
          padding: "10px 14px",
          borderRadius: 10,
          border: "1px solid #333",
          cursor: canPredict ? "pointer" : "not-allowed",
        }}
      >
        Predict
      </button>

      {result && <div style={{ marginTop: 18, fontSize: 18, fontWeight: 800 }}>{result}</div>}
    </div>
  );
}
