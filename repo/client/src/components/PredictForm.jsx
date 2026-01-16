import { useEffect, useMemo, useState } from "react";

export default function Predict() {
  const API = import.meta.env.DEV ? import.meta.env.VITE_API_BASE_URL : "";

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
        const res = await fetch(`/model-info`);
        console.log("Fetching model info from", `/model-info`);
        const info = await res.json();

        const fList = info.selected_features || [];
        setFeatures(fList);
        setRanges(info.ranges || {});
        setLabels(info.labels || {});
        setHelptext(info.helptext || {});
        setPresets(info.presets || {});
        setNote(info.note || "");

        // Initialize values to benign_typical preset if available, else midpoint
        const init = {};
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
        setError("");
      } catch (e) {
        setError("Failed to load model info. Is the Flask server running?");
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

      const res = await fetch(`/predict`, {
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
    <div className="min-h-screen bg-[#646cff]/30 text-zinc-50">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 backdrop-blur-sm p-6 shadow-[#646cff] shadow-xl">
          <div className="flex flex-col">
            <h1 className="text-2xl font-semibold tracking-tight">Breast Tumor Classifier</h1>
            <p className="text-sm text-zinc-300">
              Use presets and sliders to test the model without knowing clinical feature definitions.
            </p>
            {note ? (
              <p className="text-xs text-zinc-400">{note}</p>
            ) : null}
          </div>

          <div className="mt-6 flex flex-row space-x-3">
            <button
              onClick={() => applyPreset("benign_typical")}
              disabled={!presets.benign_typical}
              className=" rounded-full border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Typical Benign
            </button>
            <button
              onClick={() => applyPreset("malignant_typical")}
              disabled={!presets.malignant_typical}
              className=" rounded-full border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Typical Malignant
            </button>
            <button
              onClick={() => applyPreset("benign_sample")}
              disabled={!presets.benign_sample}
              className=" rounded-full border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Random Benign Sample
            </button>
            <button
              onClick={() => applyPreset("malignant_sample")}
              disabled={!presets.malignant_sample}
              className=" rounded-full border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Random Malignant Sample
            </button>
          </div>

          {error ? (
            <div className="mt-5 rounded-xl border border-red-900/50 bg-red-950/40 px-4 py-3 text-sm text-red-200">
              {error}
            </div>
          ) : null}

          <div className="mt-6 space-y-4">
            {features.map((f) => {
              const min = ranges?.[f]?.min ?? 1;
              const max = ranges?.[f]?.max ?? 10;
              const label = labels?.[f] || f;
              const help = helptext?.[f] || "";
              const value = values?.[f] ?? Math.round((min + max) / 2);

              return (
                <div
                  key={f}
                  className="rounded-2xl border border-zinc-800 bg-zinc-950/30 p-4"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex flex-col gap-1 items-start">
                      <div className="text-sm font-semibold text-zinc-100">{label}</div>
                      {help ? <div className="mt-1 text-xs text-zinc-400">{help}</div> : null}
                    </div>

                    <div className="rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-1 text-sm font-semibold text-zinc-100">
                      {value}
                    </div>
                  </div>

                  <div className="mt-3">
                    <input
                      type="range"
                      min={min}
                      max={max}
                      step="1"
                      value={value}
                      onChange={(e) =>
                        setValues((prev) => ({ ...prev, [f]: Number(e.target.value) }))
                      }
                      className="w-full accent-[#646cff] "
                    />

                    <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
                      <span>{min}</span>
                      <span>{max}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button
              onClick={onPredict}
              disabled={!canPredict}
              className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-zinc-900 hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Predict
            </button>

            {result ? (
              <div className="text-sm font-semibold text-zinc-100">{result}</div>
            ) : (
              <div className="text-sm text-zinc-400">
                Choose a preset or adjust sliders, then click Predict.
              </div>
            )}
          </div>
        </div>
      </div>
      <p className="text-center text-sm text-zinc-500">Created by John Padworski for ANA680</p>
    </div>
  );
}
